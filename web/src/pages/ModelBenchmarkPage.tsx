import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  ShieldAlert,
  FlaskConical,
  RefreshCw,
  Settings2,
  Check,
  X,
  History,
} from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { HERMES_BASE_PATH, fetchJSON } from "@/lib/api";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn } from "@/lib/utils";

// ── Types (mirror scripts/model_benchmark/build_benchmark.py output) ───────
type Dims = {
  intelligence: number;
  coding: number;
  cost: number;
  context: number;
  tool_use: number;
  speed: number;
  vision: number;
  privacy: number;
};

interface ModelRow {
  id: string;
  family: string;
  router: string | null;
  channel: string;
  billing: string;
  cost_tier: string | null;
  available: boolean;
  availability_note: string;
  output_authority: string;
  prc: boolean;
  trains_on_input: boolean;
  vision: boolean;
  context_window: number | null;
  max_output: number | null;
  cost_in: number | null;
  cost_out: number | null;
  cost_blend: number | null;
  cost_estimated: boolean;
  intelligence_raw: number | null;
  intelligence_estimated: boolean;
  intelligence_source?: string | null;
  speed_estimated: boolean;
  measured: { latency_median_s: number | null; throughput_tps: number | null; runs?: number } | null;
  arena?: { elo: number | null; seed_elo: number | null; matches: number; wins: number; settled: boolean } | null;
  privacy_label: string;
  tool_tier: string;
  strong: string;
  weak: string;
  dims: Dims;
}

interface RankRow {
  model: string;
  router: string | null;
  family: string;
  score: number;
  policy_score: number;
  policy_penalty: number | null;
  available: boolean;
  estimated: boolean;
  below_primary_floor?: boolean;
  dims: Dims;
}

interface Lane {
  key: string;
  label: string;
  category: string;
  blurb: string;
  output_authority: string;
  sensitive: boolean;
  current_primary: string;
  current_primary_display: string | null;
  current_fallbacks: string[];
  current_fallbacks_display: (string | null)[];
  current_rank: number | null;
  weights: Record<string, number>;
  primary_min_intel: number | null;
  recommended: RankRow | null;
  fallbacks: RankRow[];
  verdict: string;
  near_optimal?: boolean;
  ranking: RankRow[];
}

interface Benchmark {
  meta: {
    generated_at: string;
    source: string;
    method: string;
    model_count: number;
    lane_count: number;
    dimensions: string[];
    disclaimer: string;
    weight_mode?: string;
    vault_drift_check?: string[];
    served_from?: string;
    served_mtime?: string;
    aa_index?: { version: string; as_of: string; source: string; methodology?: Record<string, unknown>; dim_norm?: { lo: number; hi: number } };
    arena?: { judge?: string; lanes?: string[]; matches_total?: number; lmarena_asof?: string } | null;
  };
  models: ModelRow[];
  lanes: Lane[];
}

interface ModelMeta {
  channel: string;
  billing: string;
  costTier: string | null;
  costBlend: number | null;
  prc: boolean;
  trains: boolean;
}

// Live wiring read from config.yaml via /api/lane/current (not the snapshot).
interface LiveLane {
  settable: boolean;
  primary: string | null;
  fallbacks: string[];
  target: string;
}

const DIM_LABELS: Record<keyof Dims, string> = {
  intelligence: "Intel",
  coding: "Coding",
  cost: "Cost",
  context: "Context",
  tool_use: "Tools",
  speed: "Speed",
  vision: "Vision",
  privacy: "Privacy",
};
const DIM_KEYS = Object.keys(DIM_LABELS) as (keyof Dims)[];

// Short labels so the columns never collide at half-card width.
const DIM_ABBR: Record<keyof Dims, string> = {
  intelligence: "Intel",
  coding: "Code",
  cost: "Cost",
  context: "Ctx",
  tool_use: "Tools",
  speed: "Speed",
  vision: "Vis",
  privacy: "Priv",
};

const COL_TOOLTIPS: Record<string, string> = {
  intelligence: "Capability — Artificial Analysis Intelligence Index where scored, else a documented estimate (flagged 'est'). Scaled 0–100.",
  coding: "Coding-specific capability — EST: the AA intelligence index ± a per-class delta (coding specialists up, chat/vision-only down). Not yet a measured coding benchmark (Terminal-Bench/SciCode).",
  cost: "Cheaper = higher. Inverse-log of output-weighted $/MTok (0.3·in + 0.7·out).",
  context: "Usable context window, log-scaled 100K–1.05M.",
  tool_use: "Function-calling / agentic tool-use reliability tier.",
  speed: "Latency/throughput estimate from tier & family until the live harness measures it.",
  vision: "Image-input quality (0 if the model can't accept images).",
  privacy: "Data-handling posture; OpenRouter passthrough of clean vendors is down-weighted.",
  context_window: "Raw context window in tokens.",
  cost_blend: "Output-weighted blended $/MTok (0.3·in + 0.7·out); ~ = estimated.",
};

const CATEGORY_LABELS: Record<string, string> = {
  core: "Core agent",
  "kanban-worker": "Kanban workers",
  review: "Review",
  auxiliary: "Auxiliary slots",
};

function barColor(v: number): string {
  if (v >= 75) return "bg-emerald-500";
  if (v >= 50) return "bg-amber-500";
  if (v >= 25) return "bg-orange-500";
  return "bg-rose-500";
}

// Access/billing channel → badge colour. Subscription = violet, PPU = sky,
// aggregator/credits = slate.
const CHANNEL_STYLE: Record<string, string> = {
  // Subscription / seat = green
  "Claude Code": "bg-emerald-500/15 text-emerald-600",
  "ChatGPT/Codex": "bg-emerald-500/15 text-emerald-600",
  "z.ai": "bg-emerald-500/15 text-emerald-600",
  Kimi: "bg-emerald-500/15 text-emerald-600",
  Antigravity: "bg-emerald-500/15 text-emerald-600",
  // Pay-per-use direct (Vertex, MiniMax) = red
  Vertex: "bg-rose-500/15 text-rose-600",
  MiniMax: "bg-rose-500/15 text-rose-600",
  // OpenRouter (and NIM) = blue
  OpenRouter: "bg-blue-500/15 text-blue-600",
  NIM: "bg-blue-500/15 text-blue-600",
};

const COST_TIER_RANGES =
  "$ <$0.50 · $$ $0.50–1.99 · $$$ $2.00–7.99 · $$$$ $8.00–29.99 · $$$$$ ≥$30 (blended $/MTok). Informational — not in the score.";

function costTierColor(tier: string): string {
  // $ cheapest (emerald) → $$$$$ priciest (rose).
  return (
    { 1: "text-emerald-600", 2: "text-lime-600", 3: "text-amber-600", 4: "text-orange-600", 5: "text-rose-600" }[
      tier.length
    ] ?? "text-muted-foreground"
  );
}

function CostBadge({ tier, blend }: { tier?: string | null; blend?: number | null }) {
  if (!tier) return null;
  const title = blend != null ? `~$${blend}/MTok blended · ${COST_TIER_RANGES}` : COST_TIER_RANGES;
  return (
    <span title={title} className={cn("font-mono text-[10px] font-semibold tracking-tight", costTierColor(tier))}>
      {tier}
    </span>
  );
}

function ChannelBadge({ info }: { info?: { channel: string; billing: string } }) {
  if (!info || info.channel === "—") return null;
  return (
    <span
      title={info.billing}
      className={cn(
        "rounded px-1 py-0.5 text-[9px] font-medium",
        CHANNEL_STYLE[info.channel] ?? "bg-muted text-muted-foreground",
      )}
    >
      {info.channel}
    </span>
  );
}

// Recommended-model profile: 7 columns, each STACKED (abbrev label / bar /
// number, centered) so nothing collides at half-card width.
function DimGrid({ dims }: { dims: Dims }) {
  return (
    <div className="grid grid-cols-7 gap-x-1.5">
      {DIM_KEYS.map((k) => (
        <div key={k} className="flex flex-col items-center gap-0.5" title={`${DIM_LABELS[k]}: ${Math.round(dims[k])}`}>
          <span className="text-[9px] uppercase leading-none text-muted-foreground">{DIM_ABBR[k]}</span>
          <div className="h-1.5 w-full overflow-hidden rounded bg-muted">
            <div className={cn("h-full", barColor(dims[k]))} style={{ width: `${dims[k]}%` }} />
          </div>
          <span className="text-[9px] leading-none tabular-nums text-muted-foreground">{Math.round(dims[k])}</span>
        </div>
      ))}
    </div>
  );
}

// Tight inline dimension strip for ranking rows: bar + number only (labels live
// once in the ranking header above). Centered so values never touch.
function DimBarsRow({ dims }: { dims: Dims }) {
  return (
    <div className="grid grid-cols-7 gap-x-1.5">
      {DIM_KEYS.map((k) => (
        <div key={k} className="flex flex-col items-center gap-0.5" title={`${DIM_LABELS[k]}: ${Math.round(dims[k])}`}>
          <div className="h-1.5 w-full overflow-hidden rounded-sm bg-muted">
            <div className={cn("h-full", barColor(dims[k]))} style={{ width: `${dims[k]}%` }} />
          </div>
          <span className="text-[8px] leading-none tabular-nums text-muted-foreground">{Math.round(dims[k])}</span>
        </div>
      ))}
    </div>
  );
}

function shortId(routerOrId: string | null, fallback: string): string {
  return routerOrId || fallback;
}

// Format an ISO/UTC timestamp as "June 6, 2026 7:45 PM" in US Eastern time.
function formatEastern(iso?: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const date = new Intl.DateTimeFormat("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "America/New_York",
  }).format(d);
  const time = new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
    timeZone: "America/New_York",
  }).format(d);
  return `${date} ${time} ET`;
}

// Lanes that aren't settable from this host (governed elsewhere).
const READONLY_LANES: Record<string, string> = {
  reviewer: "governed by Robin review lanes — set on the review host",
};

async function applyLane(lane: string, primary: string, fallbacks: string[]) {
  return fetchJSON<{ ok: boolean; note: string; target: string; config_path: string }>(
    "/api/lane/apply",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lane, primary, fallbacks }),
    },
  );
}

// ── Per-lane "Set primary + fallbacks" panel ───────────────────────────────
function SetPanel({ lane, onApplied }: { lane: Lane; onApplied: () => void }) {
  const choices = useMemo(() => lane.ranking.filter((r) => r.router), [lane.ranking]);
  const [primary, setPrimary] = useState<string>(
    lane.recommended?.router || choices[0]?.router || "",
  );
  const [fallbacks, setFallbacks] = useState<string[]>(
    lane.fallbacks.map((f) => f.router).filter((r): r is string => !!r),
  );
  const [status, setStatus] = useState<"idle" | "saving" | "ok" | "err">("idle");
  const [msg, setMsg] = useState("");

  function toggleFb(router: string) {
    setFallbacks((cur) =>
      cur.includes(router) ? cur.filter((r) => r !== router) : [...cur, router],
    );
  }

  async function onApply() {
    setStatus("saving");
    setMsg("");
    try {
      const res = await applyLane(lane.key, primary, fallbacks.filter((f) => f !== primary));
      setStatus("ok");
      setMsg(res.note || "Applied.");
      onApplied();
    } catch (e) {
      setStatus("err");
      setMsg(String(e));
    }
  }

  return (
    <div className="space-y-2 rounded-md border border-sky-500/40 bg-sky-500/5 p-2">
      <div className="text-[10px] uppercase tracking-wide text-sky-600">Set primary + fallbacks (writes config, new sessions only)</div>
      <label className="flex items-center gap-2 text-xs">
        <span className="w-16 shrink-0 text-muted-foreground">Primary</span>
        <select
          value={primary}
          onChange={(e) => setPrimary(e.target.value)}
          className="flex-1 rounded border bg-background px-2 py-1 text-xs"
        >
          {choices.map((c) => (
            <option key={c.router!} value={c.router!}>
              {c.model} ({c.router})
            </option>
          ))}
        </select>
      </label>
      <div className="text-xs">
        <div className="mb-1 text-muted-foreground">Fallbacks (in click order)</div>
        <div className="flex flex-wrap gap-1">
          {choices
            .filter((c) => c.router !== primary)
            .map((c) => {
              const i = fallbacks.indexOf(c.router!);
              const sel = i >= 0;
              return (
                <button
                  key={c.router!}
                  onClick={() => toggleFb(c.router!)}
                  className={cn(
                    "rounded border px-1.5 py-0.5 text-[10px]",
                    sel ? "border-sky-500 bg-sky-500/15 text-sky-700" : "border-border text-muted-foreground hover:bg-muted",
                  )}
                  title={c.router!}
                >
                  {sel && <span className="mr-1 tabular-nums">{i + 1}.</span>}
                  {c.model}
                </button>
              );
            })}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Button size="sm" onClick={onApply} disabled={!primary || status === "saving"}>
          {status === "saving" ? "Applying…" : "Apply"}
        </Button>
        {status === "ok" && (
          <span className="flex items-center gap-1 text-[11px] text-emerald-600">
            <Check className="h-3 w-3" /> {msg}
          </span>
        )}
        {status === "err" && (
          <span className="flex items-center gap-1 text-[11px] text-rose-600">
            <X className="h-3 w-3" /> {msg}
          </span>
        )}
      </div>
    </div>
  );
}

// ── Lane recommendation card ───────────────────────────────────────────────
function LaneCard({
  lane,
  canApply,
  onApplied,
  live,
  disp,
  channelOf,
}: {
  lane: Lane;
  canApply: boolean;
  onApplied: () => void;
  live?: LiveLane;
  disp: (router: string) => string;
  channelOf: (model: string) => ModelMeta | undefined;
}) {
  const [open, setOpen] = useState(false);
  const [showSet, setShowSet] = useState(false);
  const [showAllRanks, setShowAllRanks] = useState(false);
  const [justApplied, setJustApplied] = useState(false);
  const readonlyReason = READONLY_LANES[lane.key];

  function handleApplied() {
    setShowSet(false);
    setJustApplied(true);
    onApplied();
    window.setTimeout(() => setJustApplied(false), 4000);
  }

  // Prefer LIVE config wiring (from /api/lane/current) for "Wired today".
  const livePrimary = live ? (live.primary ? disp(live.primary) : "auto") : null;
  const liveFallbacks = live ? live.fallbacks.map(disp) : null;
  const rec = lane.recommended;

  // Verdict computed from the LIVE wired model vs the ranking (the backend
  // verdict is static against lanes.py and goes stale after a re-pin).
  const wiredRouter = live?.primary && live.primary !== "auto" ? live.primary : null;
  const wiredIdx = wiredRouter ? lane.ranking.findIndex((r) => r.router === wiredRouter) : -1;
  let liveVerdict = "";
  let ok = true;
  if (readonlyReason) {
    liveVerdict = "";
  } else if (!rec) {
    liveVerdict = "No eligible model for this lane under current policy.";
    ok = false;
  } else if (!wiredRouter) {
    liveVerdict = "Unpinned — resolves to the main chat model.";
  } else if (wiredIdx === 0) {
    liveVerdict = "Current wiring is the top-ranked model ✓";
  } else if (wiredIdx > 0) {
    // Tolerance band: within NEAR_OPTIMAL_TOL points of the top is co-optimal —
    // avoids "data prefers X" churn on hair-thin (often cost-driven) margins.
    // Keep in sync with NEAR_OPTIMAL_TOL in scripts/model_benchmark/build_benchmark.py.
    const NEAR_OPTIMAL_TOL = 2.5;
    const gap = Math.round((lane.ranking[0].policy_score - lane.ranking[wiredIdx].policy_score) * 10) / 10;
    if (gap <= NEAR_OPTIMAL_TOL) {
      liveVerdict = `Current wiring (${disp(wiredRouter)}) ✓ near-optimal (#${wiredIdx + 1}, within ${gap.toFixed(1)} of ${rec.model}).`;
    } else {
      liveVerdict = `Current wiring (${disp(wiredRouter)}) ranks #${wiredIdx + 1}; data prefers ${rec.model}.`;
      ok = false;
    }
  } else {
    liveVerdict = `Current wiring (${disp(wiredRouter)}) isn't in this lane's eligible set; data prefers ${rec.model}.`;
    ok = false;
  }
  const VerdictIcon = ok ? CheckCircle2 : AlertTriangle;
  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="text-sm">{lane.label}</CardTitle>
          <Badge tone="outline" className="text-xs font-medium capitalize">{lane.output_authority}</Badge>
          {lane.sensitive && (
            <Badge tone="outline" className="gap-1 text-xs font-medium capitalize text-amber-600">
              <ShieldAlert className="h-3.5 w-3.5" /> sensitive
            </Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground">{lane.blurb}</p>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-3">
          <div
            className={cn(
              "space-y-1 rounded-md border p-2 transition-colors",
              justApplied ? "border-emerald-500/60 bg-emerald-500/10" : "border-border/60",
            )}
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                Wired now {live && <span className="text-emerald-600/70">· live</span>}
              </span>
              {justApplied && (
                <span className="flex items-center gap-0.5 text-[10px] text-emerald-600">
                  <Check className="h-3 w-3" /> updated
                </span>
              )}
            </div>
            <div className="font-mono text-xs">
              {livePrimary ?? shortId(lane.current_primary_display, lane.current_primary)}
            </div>
            {(liveFallbacks ?? lane.current_fallbacks_display.filter((x): x is string => !!x)).length > 0 && (
              <div className="flex flex-wrap gap-1 pt-1">
                {(liveFallbacks ?? lane.current_fallbacks_display.filter((x): x is string => !!x)).map((f, i) => (
                  <span key={i} className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">{f}</span>
                ))}
              </div>
            )}
          </div>
          <div className="space-y-1 rounded-md border border-emerald-500/40 bg-emerald-500/5 p-2">
            <div className="text-[10px] uppercase tracking-wide text-emerald-600">Recommended primary</div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold">{rec ? rec.model : "—"}</span>
              {rec && <span className="tabular-nums text-[10px] text-muted-foreground">score {rec.policy_score}</span>}
            </div>
            {lane.fallbacks.length > 0 && (
              <div className="flex flex-wrap gap-1 pt-1">
                {lane.fallbacks.map((f) => (
                  <span key={f.model} className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                    {f.model} <span className="tabular-nums opacity-60">{f.policy_score}</span>
                  </span>
                ))}
              </div>
            )}
            {rec && (
              <div className="space-y-0.5 pt-1.5">
                <div className="text-[9px] uppercase tracking-wide text-emerald-600/70">
                  Dimension scores (every model&apos;s are in the full ranking below)
                </div>
                <DimGrid dims={rec.dims} />
              </div>
            )}
          </div>
        </div>

        {liveVerdict && (
          <div className={cn("flex items-center gap-1.5 text-xs", ok ? "text-emerald-600" : "text-amber-600")}>
            <VerdictIcon className="h-3.5 w-3.5 shrink-0" />
            <span>{liveVerdict}</span>
          </div>
        )}

        {/* Set models control */}
        {readonlyReason ? (
          <p className="flex items-center gap-1 text-[11px] text-muted-foreground">
            <ShieldAlert className="h-3 w-3" /> Not settable here — {readonlyReason}.
          </p>
        ) : (
          <div className="space-y-2">
            <button
              onClick={() => setShowSet((v) => !v)}
              disabled={!canApply}
              className={cn(
                "flex items-center gap-1 text-[11px]",
                canApply ? "text-sky-600 hover:text-sky-700" : "cursor-not-allowed text-muted-foreground",
              )}
              title={canApply ? "" : "Switch to the Latest snapshot to apply changes"}
            >
              <Settings2 className="h-3 w-3" />
              {showSet ? "Cancel" : "Set default & fallbacks"}
              {!canApply && " (latest only)"}
            </button>
            {showSet && canApply && <SetPanel lane={lane} onApplied={handleApplied} />}
          </div>
        )}

        <button
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
        >
          {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          {open ? "Hide" : "Show"} full ranking ({lane.ranking.length}) & lane weights
        </button>
        {open && (
          <div className="space-y-2">
            <div className="flex flex-wrap gap-1">
              {Object.entries(lane.weights).map(([k, v]) => (
                <span key={k} className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                  {DIM_LABELS[k as keyof Dims] ?? k}×{v}
                </span>
              ))}
            </div>
            <div className="space-y-0.5">
              {/* column header — aligns with the dims | Policy/Raw/Flags band below */}
              <div className="flex items-end gap-3 border-b pb-1 pl-5 text-[10px] uppercase text-muted-foreground">
                <div className="grid min-w-0 flex-1 grid-cols-7 gap-x-1.5">
                  {DIM_KEYS.map((k) => (
                    <span key={k} className="cursor-help text-center leading-none" title={DIM_LABELS[k]}>
                      {DIM_ABBR[k]}
                    </span>
                  ))}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <span className="w-10 cursor-help text-right underline decoration-dotted" title="Policy score = Raw × policy penalty. Recommendations rank by this (PRC ×0.88, trains-on-input ×0.55 on sensitive lanes).">Policy</span>
                  <span className="w-8 cursor-help text-right underline decoration-dotted" title="Raw score = weighted mean of the 8 dimensions for this lane, before any penalty.">Raw</span>
                  <span className="w-10 cursor-help text-center underline decoration-dotted" title="Cost tier (pay-per-use models only): $ <$0.50 … $$$$$ ≥$30 blended $/MTok.">$</span>
                  <span className="w-20 text-right">Flags</span>
                </div>
              </div>
              {lane.ranking.slice(0, showAllRanks ? lane.ranking.length : 12).map((r, i) => {
                const meta = channelOf(r.model);
                const isCurrent = r.model === lane.current_primary_display;
                return (
                  <div
                    key={r.model}
                    className={cn("space-y-1 border-b border-border/40 py-1.5", isCurrent && "rounded bg-sky-500/5 px-1")}
                  >
                    {/* line 1: rank + model (left) · channel badge (far right, above flags) */}
                    <div className="flex items-center justify-between gap-1.5">
                      <div className="flex min-w-0 items-center gap-1.5">
                        <span className="w-4 shrink-0 text-right text-[11px] tabular-nums text-muted-foreground">{i + 1}</span>
                        <span className={cn("truncate text-xs", i === 0 && "font-semibold")}>{r.model}</span>
                      </div>
                      <ChannelBadge info={meta} />
                    </div>
                    {/* line 2: dimension bars (left) · Policy / Raw / Flags (far right) */}
                    <div className="flex items-end gap-3 pl-5">
                      <div className="min-w-0 flex-1">
                        <DimBarsRow dims={r.dims} />
                      </div>
                      <div className="flex shrink-0 items-center gap-2">
                        <span className="w-10 text-right text-xs font-medium tabular-nums">{r.policy_score}</span>
                        <span className="w-8 text-right text-xs tabular-nums text-muted-foreground">{r.score}</span>
                        <span className="flex w-10 justify-center">
                          <CostBadge tier={meta?.costTier} blend={meta?.costBlend} />
                        </span>
                        <span className="flex w-20 flex-wrap justify-end gap-0.5">
                          {meta?.prc && (
                            <span title="Provider under People's Republic of China jurisdiction" className="cursor-help rounded bg-orange-500/15 px-1 text-[9px] text-orange-600">PRC</span>
                          )}
                          {meta?.trains && (
                            <span title="Provider trains on API content" className="cursor-help rounded bg-rose-500/15 px-1 text-[9px] text-rose-600">trains</span>
                          )}
                          {isCurrent && (
                            <span className="rounded bg-sky-500/15 px-1 text-[9px] text-sky-600">now</span>
                          )}
                          {r.below_primary_floor && (
                            <span title="Ranks here on weighted score but its knowledge is below this lane's floor — barred from being the recommended primary or fallback (a fast-but-weak model should not run a thinking lane)." className="cursor-help rounded bg-amber-500/15 px-1 text-[9px] text-amber-600">↓floor</span>
                          )}
                          {r.estimated && (
                            <span title="Capability and/or speed is an estimate (no published AA index / not yet measured)." className="cursor-help rounded bg-muted px-1 text-[9px] text-muted-foreground">est</span>
                          )}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            {/* Legend */}
            <div className="space-y-1 rounded-md border border-border/50 bg-muted/30 p-2 text-[10px] leading-relaxed text-muted-foreground">
              <div>
                <span className="font-medium text-foreground">Policy</span> = Raw × policy penalty —
                recommendations rank by this. <span className="font-medium text-foreground">Raw</span> =
                weighted mean of the 8 dimensions for this lane, before any penalty.
              </div>
              <div>
                <span className="font-medium text-foreground">Flags:</span>{" "}
                <span className="rounded bg-sky-500/15 px-1 text-sky-600">now</span> wired today ·{" "}
                <span className="rounded bg-muted px-1">est</span> estimated input (no AA index / speed estimate) ·{" "}
                <span className="rounded bg-amber-500/15 px-1 text-amber-600">↓floor</span> knowledge below this
                lane's floor — stays in the ranking but is barred from being the primary/fallback.
                The Policy-vs-Raw gap shows any penalty applied (PRC ×0.88, trains-on-input ×0.55 on sensitive lanes).
              </div>
              <div>
                <span className="rounded bg-orange-500/15 px-1 text-orange-600">PRC</span> = provider under{" "}
                <span className="font-medium text-foreground">People's Republic of China</span> jurisdiction
                (data-governance flag) — drives the ×0.88 penalty on sensitive lanes ·{" "}
                <span className="rounded bg-rose-500/15 px-1 text-rose-600">trains</span> = provider trains on
                API content (e.g. Kimi/Moonshot) → ×0.55 penalty.
              </div>
              <div>
                <span className="font-medium text-foreground">Channel</span> = how the model is reached &amp; billed:{" "}
                <span className="rounded bg-emerald-500/15 px-1 text-emerald-600">green</span> subscription
                (Claude Code, ChatGPT/Codex, z.ai, Kimi) ·{" "}
                <span className="rounded bg-rose-500/15 px-1 text-rose-600">red</span> pay-per-use
                (Vertex, MiniMax) ·{" "}
                <span className="rounded bg-blue-500/15 px-1 text-blue-600">blue</span> OpenRouter
                (& NIM credits). Hover a badge for details.
              </div>
              <div>
                <span className="font-medium text-foreground">Cost tier</span> (informational, not in the score):{" "}
                <span className="font-mono text-emerald-600">$</span> &lt;$0.50 ·{" "}
                <span className="font-mono text-lime-600">$$</span> $0.50–1.99 ·{" "}
                <span className="font-mono text-amber-600">$$$</span> $2.00–7.99 ·{" "}
                <span className="font-mono text-orange-600">$$$$</span> $8.00–29.99 ·{" "}
                <span className="font-mono text-rose-600">$$$$$</span> ≥$30 — blended $/MTok. Subscription /
                seat-billed models (Claude Code, Codex, z.ai, Kimi) draw on quota, so no $ tier is shown.
              </div>
            </div>
            {lane.ranking.length > 12 && (
              <button
                onClick={() => setShowAllRanks((v) => !v)}
                className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
              >
                {showAllRanks ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                {showAllRanks ? "Show top 12" : `Show all ${lane.ranking.length} eligible models`}
              </button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Model matrix (sortable) ────────────────────────────────────────────────
type SortKey = "id" | "family" | keyof Dims | "context_window" | "cost_blend";

function ModelMatrix({ models }: { models: ModelRow[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("intelligence");
  const [dir, setDir] = useState<"asc" | "desc">("desc");

  const sorted = useMemo(() => {
    const get = (m: ModelRow): number | string => {
      if (sortKey === "id" || sortKey === "family") return m[sortKey];
      if (sortKey === "context_window") return m.context_window ?? 0;
      if (sortKey === "cost_blend") return m.cost_blend ?? 0;
      return m.dims[sortKey];
    };
    return [...models].sort((a, b) => {
      const av = get(a);
      const bv = get(b);
      const cmp = typeof av === "string" ? av.localeCompare(bv as string) : (av as number) - (bv as number);
      return dir === "asc" ? cmp : -cmp;
    });
  }, [models, sortKey, dir]);

  function header(key: SortKey, label: string, align: "left" | "right" = "right") {
    const active = sortKey === key;
    const Icon = !active ? ArrowUpDown : dir === "asc" ? ArrowUp : ArrowDown;
    return (
      <th
        title={COL_TOOLTIPS[key]}
        className={cn("cursor-pointer select-none whitespace-nowrap px-2 py-1.5 text-[10px] uppercase text-muted-foreground hover:text-foreground", COL_TOOLTIPS[key] && "underline decoration-dotted underline-offset-2", align === "right" ? "text-right" : "text-left")}
        onClick={() => {
          if (active) setDir((d) => (d === "asc" ? "desc" : "asc"));
          else {
            setSortKey(key);
            setDir(key === "id" || key === "family" ? "asc" : "desc");
          }
        }}
      >
        <span className={cn("inline-flex items-center gap-1", align === "right" && "flex-row-reverse")}>
          {label} <Icon className="h-3 w-3 opacity-60" />
        </span>
      </th>
    );
  }

  return (
    <div className="overflow-x-auto rounded-md border">
      <table className="w-full text-xs">
        <thead className="bg-muted/40">
          <tr>
            {header("id", "Model", "left")}
            {header("family", "Family", "left")}
            {DIM_KEYS.map((k) => header(k, DIM_LABELS[k]))}
            {header("context_window", "Ctx")}
            {header("cost_blend", "$/MTok*")}
            <th className="px-2 py-1.5 text-left text-[10px] uppercase text-muted-foreground">Flags</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((m) => (
            <tr key={m.id} className={cn("border-t border-border/40 hover:bg-muted/30", !m.available && "opacity-50")}>
              <td className="px-2 py-1.5">
                <div className="flex items-center gap-1.5">
                  <span className="font-medium">{m.id}</span>
                  <CostBadge tier={m.cost_tier} blend={m.cost_blend} />
                  <ChannelBadge info={{ channel: m.channel, billing: m.billing }} />
                </div>
                <div className="font-mono text-[10px] text-muted-foreground">{m.router ?? "—"}</div>
              </td>
              <td className="px-2 py-1.5 text-muted-foreground">{m.family}</td>
              {DIM_KEYS.map((k) => (
                <td key={k} className="px-2 py-1.5">
                  <div className="flex items-center justify-end gap-1.5">
                    <span className="tabular-nums">{Math.round(m.dims[k])}</span>
                    <div className="hidden h-1.5 w-8 overflow-hidden rounded bg-muted lg:block">
                      <div className={cn("h-full", barColor(m.dims[k]))} style={{ width: `${m.dims[k]}%` }} />
                    </div>
                  </div>
                </td>
              ))}
              <td className="px-2 py-1.5 text-right tabular-nums text-muted-foreground">
                {m.context_window ? `${Math.round(m.context_window / 1000)}K` : "—"}
              </td>
              <td className="px-2 py-1.5 text-right tabular-nums text-muted-foreground">
                {m.cost_blend != null ? `$${m.cost_blend}` : "—"}
                {m.cost_estimated && <span className="text-amber-600">~</span>}
              </td>
              <td className="px-2 py-1.5">
                <div className="flex flex-wrap gap-1">
                  {!m.available && <span className="rounded bg-rose-500/15 px-1 text-[9px] text-rose-600">unavail</span>}
                  {m.output_authority === "binding" && <span className="rounded bg-emerald-500/15 px-1 text-[9px] text-emerald-600">binding</span>}
                  {m.output_authority === "none" && <span className="rounded bg-muted px-1 text-[9px] text-muted-foreground">non-chat</span>}
                  {m.prc && <span title="Provider under People's Republic of China jurisdiction (data-governance flag)" className="cursor-help rounded bg-orange-500/15 px-1 text-[9px] text-orange-600">PRC</span>}
                  {m.trains_on_input && <span title="Provider trains on API content (e.g. Kimi/Moonshot)" className="cursor-help rounded bg-rose-500/15 px-1 text-[9px] text-rose-600">trains</span>}
                  {m.intelligence_estimated && (
                    <span title="Capability is an estimate — no published AA Intelligence Index for this model." className="cursor-help rounded bg-muted px-1 text-[9px] text-muted-foreground">intel?</span>
                  )}
                  {m.speed_estimated && (
                    <span title="Speed is an estimate — not yet measured by a live sweep (or the sweep call failed)." className="cursor-help rounded bg-muted px-1 text-[9px] text-muted-foreground">speed?</span>
                  )}
                  {m.measured && !m.speed_estimated && (
                    <span title={`Measured: ${m.measured.throughput_tps ?? "?"} tok/s, ${m.measured.latency_median_s ?? "?"}s latency`} className="cursor-help rounded bg-emerald-500/15 px-1 text-[9px] text-emerald-600">measured</span>
                  )}
                  {m.arena && m.arena.elo != null && (
                    <span
                      title={`Hermes Arena Elo ${m.arena.elo} — our own blind A/B agent eval (${m.arena.matches} matches, ${m.arena.wins} wins; seed ${m.arena.seed_elo} from LMArena/AA). ${m.arena.settled ? "settled" : "provisional (<6 matches)"}.`}
                      className="cursor-help rounded bg-violet-500/15 px-1 text-[9px] font-medium tabular-nums text-violet-600"
                    >
                      Elo {Math.round(m.arena.elo)}{!m.arena.settled && "*"}
                    </span>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────
export default function ModelBenchmarkPage() {
  const [data, setData] = useState<Benchmark | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<"lanes" | "models">("lanes");
  const [snapshots, setSnapshots] = useState<{ id: string; generated_at: string }[]>([]);
  const [selectedAt, setSelectedAt] = useState<string>(""); // "" = latest/live
  const [liveWiring, setLiveWiring] = useState<Record<string, LiveLane>>({});
  const { setTitle } = usePageHeader();

  useEffect(() => {
    setTitle("Model Benchmark");
    return () => setTitle(null);
  }, [setTitle]);

  // Load the dataset — current ("") via API (static fallback) or a dated snapshot.
  const load = useCallback(async (at: string) => {
    setLoading(true);
    setError(null);
    try {
      const j = await fetchJSON<Benchmark>(`/api/model-benchmark${at ? `?at=${encodeURIComponent(at)}` : ""}`);
      setData(j);
    } catch {
      if (at) {
        setError(`snapshot ${at} unavailable`);
      } else {
        // Static asset fallback only for the live/current dataset.
        try {
          const r = await fetch(`${HERMES_BASE_PATH}/data/model-benchmark.json`, { cache: "no-cache" });
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          setData((await r.json()) as Benchmark);
        } catch (e) {
          setError(String(e));
        }
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshHistory = useCallback(async () => {
    try {
      const h = await fetchJSON<{ snapshots: { id: string; generated_at: string }[] }>(
        "/api/model-benchmark/history",
      );
      setSnapshots(h.snapshots || []);
    } catch {
      setSnapshots([]);
    }
  }, []);

  // Live wiring (primary + fallbacks from config.yaml) for the "Wired now" box.
  const loadLiveWiring = useCallback(async () => {
    try {
      const r = await fetchJSON<{ lanes: Record<string, LiveLane> }>("/api/lane/current");
      setLiveWiring(r.lanes || {});
    } catch {
      setLiveWiring({});
    }
  }, []);

  useEffect(() => {
    load("");
    refreshHistory();
    loadLiveWiring();
  }, [load, refreshHistory, loadLiveWiring]);

  const canApply = selectedAt === "";
  function onPickSnapshot(id: string) {
    setSelectedAt(id);
    load(id);
  }

  // ── Run Benchmark (live sweep) ───────────────────────────────────────────
  const [sweep, setSweep] = useState<{
    state: string;
    done?: number;
    total?: number;
    ok?: number;
    failed?: number;
    current?: string;
    error?: string;
  }>({ state: "idle" });
  const [confirmSweep, setConfirmSweep] = useState(false);
  const sweepRunning = sweep.state === "running";

  const pollSweep = useCallback(async () => {
    try {
      const s = await fetchJSON<typeof sweep>("/api/model-benchmark/run-sweep/status");
      setSweep(s);
      return s.state;
    } catch {
      return "idle";
    }
  }, []);

  // On mount, pick up an in-flight sweep.
  useEffect(() => {
    pollSweep();
  }, [pollSweep]);

  // While running, poll every 3s; reload data when it finishes.
  useEffect(() => {
    if (!sweepRunning) return;
    const id = window.setInterval(async () => {
      const st = await pollSweep();
      if (st !== "running") {
        window.clearInterval(id);
        load(selectedAt);
        refreshHistory();
      }
    }, 3000);
    return () => window.clearInterval(id);
  }, [sweepRunning, pollSweep, load, refreshHistory, selectedAt]);

  async function startSweep() {
    setConfirmSweep(false);
    setSweep({ state: "running", done: 0, total: 0 });
    try {
      await fetchJSON("/api/model-benchmark/run-sweep", { method: "POST" });
    } catch (e) {
      setSweep({ state: "error", error: String(e) });
    }
  }

  // router id → display name, for rendering live wiring nicely.
  const disp = useMemo(() => {
    const m: Record<string, string> = {};
    data?.models.forEach((mm) => {
      if (mm.router) m[mm.router] = mm.id;
    });
    return (router: string) => m[router] ?? router;
  }, [data]);

  // model display name → channel + cost meta, for ranking-row badges.
  const channelOf = useMemo(() => {
    const m: Record<string, ModelMeta> = {};
    data?.models.forEach((mm) => {
      m[mm.id] = { channel: mm.channel, billing: mm.billing, costTier: mm.cost_tier, costBlend: mm.cost_blend, prc: mm.prc, trains: mm.trains_on_input };
    });
    return (model: string) => m[model];
  }, [data]);

  const lanesByCat = useMemo(() => {
    const m = new Map<string, Lane[]>();
    data?.lanes.forEach((l) => {
      if (!m.has(l.category)) m.set(l.category, []);
      m.get(l.category)!.push(l);
    });
    return m;
  }, [data]);

  if (loading) return <div className="flex justify-center p-12"><Spinner /></div>;
  if (error || !data)
    return (
      <div className="space-y-2 p-6 text-sm">
        <p className="text-rose-600">Could not load benchmark data: {error}</p>
        <p className="text-muted-foreground">
          Generate it with{" "}
          <code className="rounded bg-muted px-1">python3 scripts/model_benchmark/build_benchmark.py --verify</code>{" "}
          then rebuild the web bundle.
        </p>
      </div>
    );

  return (
    <div className="space-y-4 p-4">
      {/* Summary */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-sm text-muted-foreground">
            {data.meta.model_count} models × {data.meta.lane_count} lanes · data-driven scoring (+ gated live harness).
          </p>
          {(data.meta.aa_index || data.meta.arena || data.meta.weight_mode) && (
            <p className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[11px]">
              {data.meta.weight_mode && (
                <span
                  title={
                    data.meta.weight_mode === "global-flat"
                      ? "Scoring weights: GLOBAL flat override — every lane uses one capability-led vector (intelligence-dominant, cost removed), so the highest-intelligence eligible model is #1 on every lane; lanes differ only by hard constraints. Toggle off (per-lane) by unsetting HERMES_BENCHMARK_FLAT_WEIGHTS."
                      : "Scoring weights: PER-LANE — each lane scores with its own role profile (Builder→tool-use, Ops-watch→cost/speed, QA-vision→vision, …), so recommendations are role-aware. Cost counts only for metered channels; subscription/seat models are sunk-cost. Set HERMES_BENCHMARK_FLAT_WEIGHTS=1 for the flat override."
                  }
                  className={
                    "cursor-help rounded px-1.5 py-0.5 font-medium " +
                    (data.meta.weight_mode === "global-flat"
                      ? "bg-amber-500/15 text-amber-600"
                      : "bg-emerald-500/15 text-emerald-600")
                  }
                >
                  Weights: {data.meta.weight_mode}
                </span>
              )}
              {data.meta.aa_index && (
                <span
                  title={`Intelligence baseline = Artificial Analysis Intelligence Index ${data.meta.aa_index.version}, single pinned scale (as-of ${data.meta.aa_index.as_of}). Normalised ${data.meta.aa_index.dim_norm?.lo}-${data.meta.aa_index.dim_norm?.hi} → 0-100.`}
                  className="cursor-help rounded bg-indigo-500/15 px-1.5 py-0.5 font-medium text-indigo-600"
                >
                  Knowledge: AA Index {data.meta.aa_index.version}
                </span>
              )}
              {data.meta.arena && (
                <span
                  title={`Hermes Arena = our own Bradley-Terry/Elo over ${data.meta.arena.matches_total ?? 0} blind A/B lane matches, judge ${data.meta.arena.judge}, seeded from LMArena Elo (${data.meta.arena.lmarena_asof}).`}
                  className="cursor-help rounded bg-violet-500/15 px-1.5 py-0.5 font-medium text-violet-600"
                >
                  Agent Elo: Hermes Arena ({data.meta.arena.matches_total ?? 0} matches)
                </span>
              )}
            </p>
          )}
          <p className="text-xs text-muted-foreground">
            Generated {formatEastern(data.meta.generated_at)}
            {data.meta.served_mtime && data.meta.served_mtime !== data.meta.generated_at && (
              <> · refreshed {formatEastern(data.meta.served_mtime)}</>
            )}{" "}
            · source: {data.meta.source}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {/* Snapshot date/time filter */}
          <label className="flex items-center gap-1 rounded-md border bg-background px-2 py-1 text-xs">
            <History className="h-3.5 w-3.5 text-muted-foreground" />
            <select
              value={selectedAt}
              onChange={(e) => onPickSnapshot(e.target.value)}
              className="bg-background text-xs text-foreground outline-none [color-scheme:dark]"
              title="View a past benchmark snapshot"
            >
              <option className="bg-background text-foreground" value="">Latest (live)</option>
              {snapshots.map((s) => (
                <option key={s.id} value={s.id} className="bg-background text-foreground">
                  {formatEastern(s.generated_at)}
                </option>
              ))}
            </select>
          </label>
          {/* Run Benchmark (live sweep) */}
          {sweepRunning ? (
            <span className="flex items-center gap-1.5 rounded-md border border-sky-500/40 bg-sky-500/10 px-2 py-1 text-xs text-sky-600">
              <RefreshCw className="h-3.5 w-3.5 animate-spin" />
              Running sweep… {sweep.total ? `${sweep.done}/${sweep.total}` : ""}
            </span>
          ) : confirmSweep ? (
            <span className="flex items-center gap-1.5 rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-xs">
              <span className="text-amber-700">Live sweep spends ~$0.16 — run?</span>
              <button onClick={startSweep} className="rounded bg-sky-600 px-1.5 py-0.5 text-[11px] font-medium text-white">Run</button>
              <button onClick={() => setConfirmSweep(false)} className="text-[11px] text-muted-foreground hover:text-foreground">Cancel</button>
            </span>
          ) : (
            <Button size="sm" onClick={() => setConfirmSweep(true)} title="Run the live sweep (measures latency/throughput/cost across all reachable models)">
              <RefreshCw className="h-3.5 w-3.5" />
              Run Benchmark
            </Button>
          )}
          {sweep.state === "done" && (
            <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-600">
              swept {sweep.ok}✓{sweep.failed ? ` ${sweep.failed}✗` : ""}
            </span>
          )}
          {sweep.state === "error" && (
            <span className="rounded bg-rose-500/15 px-1.5 py-0.5 text-[10px] text-rose-600" title={sweep.error}>sweep error</span>
          )}
          {!canApply && (
            <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-600">
              viewing snapshot — read-only
            </span>
          )}
          {/* View toggle */}
          <div className="flex gap-1 rounded-md border p-0.5">
            <Button size="sm" ghost={view !== "lanes"} onClick={() => setView("lanes")}>
              By lane
            </Button>
            <Button size="sm" ghost={view !== "models"} onClick={() => setView("models")}>
              Model matrix
            </Button>
          </div>
        </div>
      </div>

      {/* Methodology / disclaimer */}
      <Card className="border-amber-500/30 bg-amber-500/5">
        <CardContent className="flex gap-2 p-3 text-xs text-muted-foreground">
          <FlaskConical className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
          <div className="space-y-1">
            <p>{data.meta.method}</p>
            <p className="text-amber-700/80">{data.meta.disclaimer}</p>
            {data.meta.vault_drift_check && (
              <p className="text-[11px]">vault check: {data.meta.vault_drift_check.join("; ")}</p>
            )}
          </div>
        </CardContent>
      </Card>

      {view === "lanes" ? (
        <div className="space-y-6">
          {[...lanesByCat.entries()].map(([cat, lanes]) => (
            <section key={cat} className="space-y-3">
              <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                {CATEGORY_LABELS[cat] ?? cat} <span className="opacity-60">({lanes.length})</span>
              </h2>
              <div className="grid gap-3 lg:grid-cols-2">
                {lanes.map((l) => (
                  <LaneCard
                    key={l.key}
                    lane={l}
                    canApply={canApply}
                    onApplied={loadLiveWiring}
                    live={liveWiring[l.key]}
                    disp={disp}
                    channelOf={channelOf}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <ModelMatrix models={data.models} />
      )}

      <div className="space-y-1 pt-2 text-[10px] text-muted-foreground">
        <p>
          <span className="font-medium text-foreground">Key:</span>{" "}
          <span className="rounded bg-orange-500/15 px-1 text-orange-600">PRC</span> = provider under{" "}
          <span className="font-medium text-foreground">People&apos;s Republic of China</span> jurisdiction
          (data-governance flag; ×0.88 penalty on sensitive lanes) ·{" "}
          <span className="rounded bg-rose-500/15 px-1 text-rose-600">trains</span> = provider trains on API
          content (×0.55) · channel colour:{" "}
          <span className="text-emerald-600">green</span> subscription ·{" "}
          <span className="text-rose-600">red</span> pay-per-use ·{" "}
          <span className="text-blue-600">blue</span> OpenRouter ·{" "}
          cost tier <span className="font-mono">$–$$$$$</span> shown for pay-per-use models only.
        </p>
        <p>
          * $/MTok is an output-weighted blended rate (0.3·in + 0.7·out); ~ = estimated. Dimensions are normalised 0–100.
          Recommendations rank by policy-adjusted score and are advisory only — nothing here changes routing automatically.
        </p>
      </div>
    </div>
  );
}
