/**
 * Hermes Console — Portfolio Board dashboard plugin (V1.0, read-only).
 *
 * Calls the plugin's backend at /api/plugins/hermes-console/portfolio/
 * (mounted by the dashboard plugin system) and renders:
 *
 *   - Filter bar (keyword, board, subagent role, include done/archived)
 *   - Hierarchy pane (collapsible, left side)
 *   - Backlog tab (dense table, one row per item)
 *   - Board tab (columns enumerated from canonical statuses)
 *   - Detail drawer (right-side, Hermes-style 14 sections)
 *
 * V1.0 is intentionally GET-only: every action button in the detail
 * pane is rendered as `disabled: true` and the comment composer is a
 * placeholder pointing to V1.1. No mutation API call is attempted.
 *
 * Plain IIFE, no build step. Uses window.__HERMES_PLUGIN_SDK__ for
 * React + shadcn primitives; window.__HERMES_PLUGINS__.register to
 * install the page component. Composite IDs ({board_slug}:{task_id})
 * are parsed/synthesised consistently with the backend.
 */
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK) return;

  const { React } = SDK;
  const h = React.createElement;
  const {
    Card, CardHeader, CardTitle, CardContent,
    Badge, Button, Input, Label, Select, SelectOption,
    Separator, Tabs, TabsList, TabsTrigger, TabsContent,
  } = SDK.components;
  const { useState, useEffect, useCallback, useMemo, useRef } = SDK.hooks;
  const { cn, timeAgo } = SDK.utils;

  // ---------------------------------------------------------------------
  // Constants — kept in sync with plugin_api.py
  // ---------------------------------------------------------------------

  const CANONICAL_STATUSES = [
    "triage", "todo", "scheduled", "ready", "running",
    "blocked", "review", "done", "archived",
  ];
  const VISIBLE_STATUSES = CANONICAL_STATUSES.filter(function (s) {
    return s !== "archived";
  });

  // Friendly labels for canonical statuses. Used everywhere a status
  // badge is rendered; never invent a label that disagrees with the
  // backend's enum.
  const STATUS_LABEL = {
    triage: "Triage",
    todo: "Todo",
    scheduled: "Scheduled",
    ready: "Ready",
    running: "In Progress",
    blocked: "Blocked",
    review: "Review",
    done: "Done",
    archived: "Archived",
  };

  // Statuses collapse to dark/light card surfaces. The list mirrors
  // the canonical Hermes palette so cards slot in alongside the Kanban
  // plugin without a contrast shock.
  const STATUS_TONE = {
    triage: "muted",
    todo: "muted",
    scheduled: "info",
    ready: "info",
    running: "accent",
    blocked: "warning",
    review: "accent",
    done: "ok",
    archived: "muted",
  };

  // Lifecycle options exposed as informational chips. None of these
  // have data in V1 — they surface "N/A" in the detail pane rather
  // than fake a value.
  const LIFECYCLE_LABEL = {
    intake: "Intake / Idea",
    discovery: "Discovery",
    planned: "Planned",
    building: "Building",
    mvp: "MVP",
    "demo-ready": "Demo-ready",
    "public-delivery": "Public delivery",
  };

  // ---------------------------------------------------------------------
  // Identity helpers — composite IDs are the only globally-unique handle
  // ---------------------------------------------------------------------

  function portfolioId(boardSlug, taskId) {
    if (!boardSlug || !taskId) return null;
    return boardSlug + ":" + taskId;
  }

  function parsePortfolioId(pid) {
    if (!pid || typeof pid !== "string") return null;
    const idx = pid.indexOf(":");
    if (idx <= 0) return null;
    return {
      board_slug: pid.slice(0, idx),
      task_id: pid.slice(idx + 1),
    };
  }

  // ---------------------------------------------------------------------
  // API helpers — thin wrappers around SDK.fetchJSON with the plugin
  // base path baked in. Every error string is human-readable.
  // ---------------------------------------------------------------------

  function apiBase() {
    return "/api/plugins/hermes-console/portfolio";
  }

  function apiPath(p) {
    return apiBase() + (p.startsWith("/") ? p : "/" + p);
  }

  function buildQuery(params) {
    if (!params) return "";
    const parts = [];
    Object.keys(params).forEach(function (k) {
      const v = params[k];
      if (v === undefined || v === null || v === "") return;
      if (Array.isArray(v)) {
        if (v.length === 0) return;
        parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(v.join(",")));
      } else if (typeof v === "boolean") {
        parts.push(encodeURIComponent(k) + "=" + (v ? "true" : "false"));
      } else {
        parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(String(v)));
      }
    });
    return parts.length ? "?" + parts.join("&") : "";
  }

  function fetchSummary() {
    return SDK.fetchJSON(apiPath("/summary"));
  }
  function fetchBoards() {
    return SDK.fetchJSON(apiPath("/boards"));
  }
  function fetchFilters() {
    return SDK.fetchJSON(apiPath("/filters"));
  }
  function fetchItems(params) {
    return SDK.fetchJSON(apiPath("/items") + buildQuery(params || {}));
  }
  function fetchItem(boardSlug, taskId) {
    return SDK.fetchJSON(apiPath("/item/" + encodeURIComponent(boardSlug) + "/" + encodeURIComponent(taskId)));
  }
  function fetchEvents(params) {
    return SDK.fetchJSON(apiPath("/events") + buildQuery(params || {}));
  }

  // ---------------------------------------------------------------------
  // UI helpers
  // ---------------------------------------------------------------------

  function StatusBadge(props) {
    const status = props.status;
    if (!status) return null;
    const tone = STATUS_TONE[status] || "muted";
    const label = STATUS_LABEL[status] || status;
    return h(Badge, {
      variant: tone,
      className: "hermes-console-status-badge",
      title: "Status: " + label,
    }, label);
  }

  function PriorityChip(props) {
    const p = Number(props.priority || 0);
    if (p === 0) return null;
    const label = p > 0 ? "↑" + p : "↓" + (-p);
    return h(Badge, {
      variant: "outline",
      className: "hermes-console-priority-chip",
      title: "Priority: " + p,
    }, label);
  }

  function StaleFlag(props) {
    if (!props.stale) return null;
    return h(Badge, {
      variant: "warning",
      className: "hermes-console-stale-flag",
      title: "No activity in 24+ hours",
    }, "stale");
  }

  function NeedsKeithFlag(props) {
    if (!props.needsKeith) return null;
    return h(Badge, {
      variant: "warning",
      className: "hermes-console-keith-flag",
      title: "This task is waiting on Keith for an approval / decision",
    }, "needs keith");
  }

  // Hermes-style collapsible section used inside the detail pane.
  // Mirrors the kanban plugin's section-head/section-toggle pattern so
  // operators get the same visual language across both surfaces.
  function HermesSection(props) {
    const { title, defaultOpen, children, meta } = props;
    const initial = defaultOpen !== false;
    const [open, setOpen] = useState(initial);
    if (!children) return null;
    return h("section", {
      className: "hermes-console-section" + (open ? " hermes-console-section--open" : " hermes-console-section--collapsed"),
    },
      h("header", {
        className: "hermes-console-section-head",
        onClick: function () { setOpen(!open); },
        role: "button",
        tabIndex: 0,
        onKeyDown: function (e) {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpen(!open); }
        },
      },
        h("span", { className: "hermes-console-section-toggle" }, open ? "▾" : "▸"),
        h("span", { className: "hermes-console-section-title" }, title),
        meta ? h("span", { className: "hermes-console-section-meta" }, meta) : null
      ),
      open ? h("div", { className: "hermes-console-section-body" }, children) : null
    );
  }

  // Metadata table — same shape as the kanban TaskDetail metadata
  // rows, so the two drawers are visually consistent.
  function MetadataTable(props) {
    const rows = props.rows || [];
    if (rows.length === 0) return null;
    return h("table", { className: "hermes-console-meta-table" },
      h("tbody", null,
        rows.map(function (r, i) {
          return h("tr", { key: i },
            h("th", { scope: "row" }, r.label),
            h("td", null, r.value)
          );
        })
      )
    );
  }

  // Disabled action button — V1.0 renders every workflow action as a
  // disabled placeholder. The button still announces its intent so
  // operators see what V1.1 will unlock.
  function DisabledAction(props) {
    return h(Button, {
      variant: "outline",
      size: "sm",
      disabled: true,
      title: "Coming in V1.1 — Portfolio read-only",
      className: "hermes-console-action-disabled",
    }, props.label);
  }

  // ---------------------------------------------------------------------
  // Filter bar — keyword, board, subagent role, include done/archived
  // ---------------------------------------------------------------------

  function FilterBar(props) {
    const filters = props.filters;
    const setFilters = props.setFilters;
    const facets = props.facets || {};
    const onReset = props.onReset;

    const set = useCallback(function (patch) {
      setFilters(Object.assign({}, filters, patch));
    }, [filters, setFilters]);

    function onKeywordChange(e) {
      set({ q: e.target.value });
    }
    function onBoardChange(e) {
      set({ boards: e.target.value });
    }
    function onAssigneeChange(e) {
      set({ assignees: e.target.value });
    }
    function onIncludeDoneChange(e) {
      set({ include_done: e.target.checked });
    }
    function onIncludeArchivedChange(e) {
      set({ include_archived: e.target.checked });
    }

    const boardValues = (facets.boards || []).map(function (b) {
      return h(SelectOption, { key: b.value, value: b.value },
        b.value + " (" + b.count + ")"
      );
    });
    const assigneeValues = (facets.assignees || []).map(function (a) {
      return h(SelectOption, { key: a.value, value: a.value },
        a.value + " (" + a.count + ")"
      );
    });

    return h("div", { className: "hermes-console-filterbar" },
      h("div", { className: "hermes-console-filterbar-row" },
        h(Label, { htmlFor: "hc-filter-q" }, "Search"),
        h(Input, {
          id: "hc-filter-q",
          type: "search",
          placeholder: "title or body…",
          value: filters.q || "",
          onChange: onKeywordChange,
          className: "hermes-console-filter-input",
        })
      ),
      h("div", { className: "hermes-console-filterbar-row" },
        h(Label, { htmlFor: "hc-filter-board" }, "Board"),
        h(Select, {
          id: "hc-filter-board",
          value: filters.boards || "",
          onValueChange: onBoardChange,
          placeholder: "All boards",
        },
          h(SelectOption, { value: "" }, "All boards"),
          boardValues
        )
      ),
      h("div", { className: "hermes-console-filterbar-row" },
        h(Label, { htmlFor: "hc-filter-role" }, "Subagent role"),
        h(Select, {
          id: "hc-filter-role",
          value: filters.assignees || "",
          onValueChange: onAssigneeChange,
          placeholder: "All roles",
        },
          h(SelectOption, { value: "" }, "All roles"),
          assigneeValues
        )
      ),
      h("div", { className: "hermes-console-filterbar-row hermes-console-filterbar-checks" },
        h(Label, { className: "hermes-console-inline-label" },
          h("input", {
            type: "checkbox",
            checked: !!filters.include_done,
            onChange: onIncludeDoneChange,
          }),
          " Include done"
        ),
        h(Label, { className: "hermes-console-inline-label" },
          h("input", {
            type: "checkbox",
            checked: !!filters.include_archived,
            onChange: onIncludeArchivedChange,
          }),
          " Include archived"
        ),
        h(Button, {
          variant: "ghost",
          size: "sm",
          onClick: onReset,
          className: "hermes-console-filter-reset",
        }, "Reset")
      )
    );
  }

  // ---------------------------------------------------------------------
  // Hierarchy pane — collapsible left rail with work-item-type buckets
  // and a parent/child tree. For V1 the tree is best-effort (uses
  // the backend's hierarchy_path for the selected item).
  // ---------------------------------------------------------------------

  function HierarchyPane(props) {
    const items = props.items || [];
    const open = props.open !== false;
    const onToggle = props.onToggle;
    const selectedId = props.selectedId;
    const onSelect = props.onSelect;

    // Bucket items by work_item_type. Today every item is
    // "unclassified" (no metadata column yet) — the bucket still
    // renders, with a count, so operators see the unclassified
    // status honestly rather than as a hidden bug.
    const buckets = useMemo(function () {
      const map = {};
      items.forEach(function (it) {
        const t = it.work_item_type || "unclassified";
        map[t] = (map[t] || 0) + 1;
      });
      return Object.keys(map).sort().map(function (k) {
        return { key: k, count: map[k] };
      });
    }, [items]);

    return h("aside", {
      className: "hermes-console-hierarchy" + (open ? "" : " hermes-console-hierarchy--collapsed"),
    },
      h("header", {
        className: "hermes-console-hierarchy-head",
        onClick: onToggle,
        role: "button",
        tabIndex: 0,
        onKeyDown: function (e) {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onToggle && onToggle(); }
        },
      },
        h("span", { className: "hermes-console-hierarchy-toggle" }, open ? "▾" : "▸"),
        h("span", null, "Hierarchy")
      ),
      open ? h("div", { className: "hermes-console-hierarchy-body" },
        h("div", { className: "hermes-console-hierarchy-help" },
          "V1 has no typed project/epic/feature metadata. Existing cards appear as ",
          h("strong", null, "unclassified"),
          "."
        ),
        h("ul", { className: "hermes-console-hierarchy-list" },
          buckets.map(function (b) {
            return h("li", { key: b.key, className: "hermes-console-hierarchy-bucket" },
              h("span", { className: "hermes-console-hierarchy-name" }, b.key),
              h("span", { className: "hermes-console-hierarchy-count" }, b.count)
            );
          })
        ),
        selectedId ? h("div", { className: "hermes-console-hierarchy-selected" },
          h("div", { className: "hermes-console-hierarchy-selected-label" }, "Selected"),
          h("div", { className: "hermes-console-hierarchy-selected-id", title: selectedId },
            selectedId.length > 24 ? selectedId.slice(0, 22) + "…" : selectedId
          )
        ) : null
      ) : null
    );
  }

  // ---------------------------------------------------------------------
  // Backlog table — dense list of items.
  // ---------------------------------------------------------------------

  function BacklogTable(props) {
    const items = props.items || [];
    const onSelect = props.onSelect;
    const selectedId = props.selectedId;
    if (items.length === 0) {
      return h("div", { className: "hermes-console-empty" },
        "No items match the current filters."
      );
    }
    return h("div", { className: "hermes-console-backlog-wrap" },
      h("table", { className: "hermes-console-backlog" },
        h("thead", null,
          h("tr", null,
            h("th", null, "Status"),
            h("th", null, "Title"),
            h("th", null, "Board"),
            h("th", null, "Role"),
            h("th", null, "Priority"),
            h("th", null, "Updated"),
            h("th", null, "Flags")
          )
        ),
        h("tbody", null,
          items.map(function (it) {
            const pid = it.portfolio_id;
            const isSelected = pid === selectedId;
            return h("tr", {
              key: pid,
              className: "hermes-console-backlog-row" + (isSelected ? " hermes-console-backlog-row--selected" : ""),
              onClick: function () { onSelect && onSelect(it); },
            },
              h("td", null, h(StatusBadge, { status: it.status })),
              h("td", { className: "hermes-console-backlog-title" },
                h("span", { className: "hermes-console-backlog-title-text" }, it.title || "(untitled)"),
                it.body_preview ? h("div", { className: "hermes-console-backlog-body-preview" }, it.body_preview) : null
              ),
              h("td", null, h("code", { className: "hermes-console-board-slug" }, it.board_slug)),
              h("td", null, it.subagent_role || h("span", { className: "hermes-console-muted" }, "—")),
              h("td", null, h(PriorityChip, { priority: it.priority })),
              h("td", { title: new Date((it.updated_at || 0) * 1000).toISOString() },
                it.updated_at ? timeAgo(it.updated_at * 1000) : h("span", { className: "hermes-console-muted" }, "—")
              ),
              h("td", { className: "hermes-console-backlog-flags" },
                h(StaleFlag, { stale: it.stale }),
                h(NeedsKeithFlag, { needsKeith: it.needs_keith })
              )
            );
          })
        )
      )
    );
  }

  // ---------------------------------------------------------------------
  // Board view — Kanban columns enumerated from canonical statuses.
  // ---------------------------------------------------------------------

  function BoardView(props) {
    const byStatus = props.byStatus || {};
    const onSelect = props.onSelect;
    const selectedId = props.selectedId;
    const totalCount = VISIBLE_STATUSES.reduce(function (acc, s) {
      return acc + ((byStatus[s] || []).length);
    }, 0);
    if (totalCount === 0) {
      return h("div", { className: "hermes-console-empty" },
        "No items match the current filters."
      );
    }
    return h("div", { className: "hermes-console-board" },
      VISIBLE_STATUSES.map(function (status) {
        const cards = byStatus[status] || [];
        return h("div", {
          key: status,
          className: "hermes-console-board-col",
        },
          h("header", { className: "hermes-console-board-col-head" },
            h(StatusBadge, { status: status }),
            h("span", { className: "hermes-console-board-col-count" }, cards.length)
          ),
          h("div", { className: "hermes-console-board-col-body" },
            cards.length === 0
              ? h("div", { className: "hermes-console-board-col-empty" }, "—")
              : cards.map(function (it) {
                const pid = it.portfolio_id;
                const isSelected = pid === selectedId;
                return h("div", {
                  key: pid,
                  className: "hermes-console-card" + (isSelected ? " hermes-console-card--selected" : ""),
                  onClick: function () { onSelect && onSelect(it); },
                },
                  h("div", { className: "hermes-console-card-title" }, it.title || "(untitled)"),
                  h("div", { className: "hermes-console-card-meta" },
                    h("code", { className: "hermes-console-board-slug" }, it.board_slug),
                    it.subagent_role ? h("span", { className: "hermes-console-card-role" }, "@" + it.subagent_role) : null
                  ),
                  h("div", { className: "hermes-console-card-flags" },
                    h(PriorityChip, { priority: it.priority }),
                    h(StaleFlag, { stale: it.stale }),
                    h(NeedsKeithFlag, { needsKeith: it.needs_keith })
                  ),
                  it.body_preview ? h("div", { className: "hermes-console-card-preview" }, it.body_preview) : null
                );
              })
          )
        );
      })
    );
  }

  // ---------------------------------------------------------------------
  // Detail drawer — Hermes-style 14 sections.
  // ---------------------------------------------------------------------

  function DetailDrawer(props) {
    const item = props.item;
    const loading = props.loading;
    const error = props.error;
    const onClose = props.onClose;
    if (!item) {
      return h("aside", { className: "hermes-console-detail hermes-console-detail--empty" },
        h("div", { className: "hermes-console-detail-empty" },
          loading
            ? h("span", null, "Loading…")
            : (error || "Select an item to see its detail.")
        )
      );
    }
    const meta = item.metadata || {};
    const rows = [
      { label: "Status", value: h(StatusBadge, { status: item.task && item.task.status }) },
      { label: "Board", value: h("code", null, item.board_slug) },
      { label: "Task id", value: h("code", null, item.task_id) },
      { label: "Assignee", value: item.task && item.task.assignee ? h("code", null, item.task.assignee) : h("span", { className: "hermes-console-muted" }, "unassigned") },
      { label: "Priority", value: item.task ? h(PriorityChip, { priority: item.task.priority }) : null },
      { label: "Created by", value: item.task && item.task.created_by ? item.task.created_by : h("span", { className: "hermes-console-muted" }, "—") },
      { label: "Created at", value: item.task && item.task.created_at ? new Date(item.task.created_at * 1000).toLocaleString() : h("span", { className: "hermes-console-muted" }, "—") },
      { label: "Updated", value: item.task && item.task.updated_at ? timeAgo(item.task.updated_at * 1000) : h("span", { className: "hermes-console-muted" }, "—") },
      { label: "Completed at", value: item.task && item.task.completed_at ? new Date(item.task.completed_at * 1000).toLocaleString() : h("span", { className: "hermes-console-muted" }, "—") },
      { label: "Tenant", value: item.task && item.task.tenant ? h("code", null, item.task.tenant) : h("span", { className: "hermes-console-muted" }, "—") },
      { label: "Work item type", value: meta.work_item_type ? h(Badge, { variant: "outline" }, meta.work_item_type) : h("span", { className: "hermes-console-muted" }, "unclassified") },
      { label: "Lifecycle", value: meta.lifecycle_state ? LIFECYCLE_LABEL[meta.lifecycle_state] || meta.lifecycle_state : h("span", { className: "hermes-console-muted" }, "Not set") },
      { label: "Agent profile", value: meta.agent_profile ? meta.agent_profile : h("span", { className: "hermes-console-muted" }, "Not set") },
      { label: "Tags", value: (meta.tags || []).length === 0 ? h("span", { className: "hermes-console-muted" }, "None") : h("div", { className: "hermes-console-tag-row" }, meta.tags.map(function (t) { return h(Badge, { key: t, variant: "outline" }, t); })) },
      { label: "Progress", value: meta.progress ? (meta.progress.completed + "/" + meta.progress.total + " (" + meta.progress.percent + "%)") : h("span", { className: "hermes-console-muted" }, "N/A") },
      { label: "Needs Keith", value: meta.needs_keith ? h(NeedsKeithFlag, { needsKeith: true }) : h("span", { className: "hermes-console-muted" }, "no") },
      { label: "Stale", value: meta.stale ? h(StaleFlag, { stale: true }) : h("span", { className: "hermes-console-muted" }, "no") },
    ];

    return h("aside", { className: "hermes-console-detail" },
      h("header", { className: "hermes-console-detail-head" },
        h("div", { className: "hermes-console-detail-id", title: item.portfolio_id }, item.portfolio_id),
        h(Button, { variant: "ghost", size: "sm", onClick: onClose }, "Close")
      ),
      // 1. id/header
      h(HermesSection, { title: "Header", defaultOpen: true },
        h("h2", { className: "hermes-console-detail-title" }, item.task && item.task.title || "(untitled)")
      ),
      // 2. title (rendered inside the header above as a h2)
      // 3. metadata table
      h(HermesSection, { title: "Metadata", defaultOpen: true, meta: (rows.length + " fields") },
        h(MetadataTable, { rows: rows })
      ),
      // 4. workflow actions — all disabled in V1.0
      h(HermesSection, { title: "Actions", defaultOpen: false },
        h("div", { className: "hermes-console-action-row" },
          h(DisabledAction, { label: "Edit" }),
          h(DisabledAction, { label: "Reassign" }),
          h(DisabledAction, { label: "Move status" }),
          h(DisabledAction, { label: "Block" }),
          h(DisabledAction, { label: "Archive" })
        ),
        h("p", { className: "hermes-console-action-note" },
          "Portfolio Board V1.0 is read-only. Mutation endpoints land in V1.1."
        )
      ),
      // 5. notification / home-channel status
      h(HermesSection, { title: "Notifications", defaultOpen: false },
        h("p", { className: "hermes-console-muted" },
          "Home-channel subscriptions are managed by the native Kanban dashboard. Portfolio V1.0 surfaces no extra controls."
        )
      ),
      // 6. approval section
      h(HermesSection, { title: "Approvals", defaultOpen: true,
        meta: (item.approvals || []).length === 0 ? "none" : ((item.approvals || []).length + " pending")
      },
        (item.approvals || []).length === 0
          ? h("p", { className: "hermes-console-muted" },
              meta.needs_keith
                ? "This task looks like it needs a Keith decision, but no structured approval has been recorded yet (V1.0 placeholder)."
                : "No approval artifacts on this task."
            )
          : h("ul", { className: "hermes-console-approvals" },
              item.approvals.map(function (ap) {
                return h("li", { key: ap.approval_id || ap.id, className: "hermes-console-approval" },
                  h("div", { className: "hermes-console-approval-type" }, ap.approval_type || "approval"),
                  h("div", { className: "hermes-console-approval-status" }, ap.status)
                );
              })
            )
      ),
      // 7. description
      h(HermesSection, { title: "Description", defaultOpen: true },
        item.task && item.task.body_preview
          ? h("pre", { className: "hermes-console-description" }, item.task.body_preview)
          : h("p", { className: "hermes-console-muted" }, "No description.")
      ),
      // 8. dependencies
      h(HermesSection, {
        title: "Dependencies",
        defaultOpen: false,
        meta: ((item.links || {}).parents || []).length + " parents / " + ((item.links || {}).children || []).length + " children",
      },
        renderLinks((item.links || {}).parents || [], "Parents"),
        renderLinks((item.links || {}).children || [], "Children"),
        (item.metadata || {}).hierarchy_path && item.metadata.hierarchy_path.length > 0
          ? h("div", { className: "hermes-console-hierarchy-path" },
              h("h4", null, "Hierarchy path"),
              h("ol", null,
                item.metadata.hierarchy_path.map(function (p, i) {
                  return h("li", { key: i },
                    h("code", null, p.portfolio_id)
                  );
                })
              )
            )
          : null
      ),
      // 9. attachments
      h(HermesSection, {
        title: "Attachments",
        defaultOpen: false,
        meta: (item.attachments || []).length === 0 ? "none" : (item.attachments.length + " files"),
      },
        (item.attachments || []).length === 0
          ? h("p", { className: "hermes-console-muted" }, "No attachments.")
          : h("ul", { className: "hermes-console-attachments" },
              item.attachments.map(function (a) {
                return h("li", { key: a.id, className: "hermes-console-attachment" },
                  h("span", { className: "hermes-console-attachment-name" }, a.filename),
                  h("span", { className: "hermes-console-attachment-meta" },
                    a.content_type || "?",
                    " · ",
                    humanSize(a.size)
                  )
                );
              })
            )
      ),
      // 10. comments
      h(HermesSection, {
        title: "Comments",
        defaultOpen: false,
        meta: (item.comments || []).length === 0 ? "none" : (item.comments.length + " comments"),
      },
        (item.comments || []).length === 0
          ? h("p", { className: "hermes-console-muted" }, "No comments yet.")
          : h("ul", { className: "hermes-console-comments" },
              item.comments.map(function (c) {
                return h("li", { key: c.id, className: "hermes-console-comment" },
                  h("div", { className: "hermes-console-comment-head" },
                    h("span", { className: "hermes-console-comment-author" }, c.author || "unknown"),
                    h("span", { className: "hermes-console-comment-when" },
                      c.created_at ? timeAgo(c.created_at * 1000) : ""
                    )
                  ),
                  c.redacted
                    ? h("div", { className: "hermes-console-comment-body hermes-console-comment-body--redacted" },
                        "[body redacted — may have contained secrets]"
                      )
                    : h("pre", { className: "hermes-console-comment-body" }, c.body_preview || "")
                );
              })
            )
      ),
      // 11. events
      h(HermesSection, {
        title: "Events",
        defaultOpen: false,
        meta: (item.events || []).length === 0 ? "none" : (item.events.length + " events"),
      },
        (item.events || []).length === 0
          ? h("p", { className: "hermes-console-muted" }, "No events recorded.")
          : h("ol", { className: "hermes-console-events" },
              item.events.slice().reverse().map(function (e) {
                return h("li", { key: e.event_id, className: "hermes-console-event" },
                  h("span", { className: "hermes-console-event-kind" }, e.kind),
                  h("span", { className: "hermes-console-event-when" },
                    e.created_at ? timeAgo(e.created_at * 1000) : ""
                  ),
                  e.payload
                    ? h("pre", { className: "hermes-console-event-payload" },
                        typeof e.payload === "string" ? e.payload : JSON.stringify(e.payload)
                      )
                    : null
                );
              })
            )
      ),
      // 12. worker log — collapsed placeholder. V1.0 deliberately
      // does not surface raw worker log content.
      h(HermesSection, { title: "Worker log", defaultOpen: false },
        h("p", { className: "hermes-console-muted" },
          "Worker log content is hidden in Portfolio V1.0. Use the native Kanban task drawer for raw log access."
        )
      ),
      // 13. run history
      h(HermesSection, {
        title: "Run history",
        defaultOpen: false,
        meta: (item.runs || []).length === 0 ? "none" : (item.runs.length + " runs"),
      },
        (item.runs || []).length === 0
          ? h("p", { className: "hermes-console-muted" }, "No runs recorded.")
          : h("ol", { className: "hermes-console-runs" },
              item.runs.slice().reverse().map(function (r) {
                return h("li", { key: r.id, className: "hermes-console-run" },
                  h("div", { className: "hermes-console-run-head" },
                    h("span", { className: "hermes-console-run-profile" }, r.profile || "—"),
                    h(Badge, { variant: r.outcome === "completed" ? "ok" : "muted" }, r.outcome || r.status || "—")
                  ),
                  r.summary_preview
                    ? h("pre", { className: "hermes-console-run-summary" }, r.summary_preview)
                    : null,
                  r.error_preview
                    ? h("pre", { className: "hermes-console-run-error" }, r.error_preview)
                    : null
                );
              })
            )
      ),
      // 14. comment composer — disabled placeholder
      h(HermesSection, { title: "Comment composer", defaultOpen: false },
        h("div", { className: "hermes-console-composer-placeholder" },
          h("p", { className: "hermes-console-muted" },
            "Posting comments from Portfolio lands in V1.1. For now, use the native Kanban task drawer."
          ),
          h(Button, { variant: "outline", size: "sm", disabled: true }, "Add comment (coming in V1.1)")
        )
      ),
      // Security footer — surfaces redaction scope so operators
      // understand why some fields are "hidden".
      h("footer", { className: "hermes-console-detail-footer" },
        h("div", null,
          h("strong", null, "Security:"),
          " sensitive fields redacted — ",
          h("code", null, (item.security && item.security.hidden_fields || []).join(", "))
        )
      )
    );
  }

  function renderLinks(list, title) {
    if (!list || list.length === 0) return null;
    return h("div", { className: "hermes-console-link-group" },
      h("h4", null, title),
      h("ul", null,
        list.map(function (l) {
          return h("li", { key: l.task_id },
            h("code", null, l.portfolio_id || l.task_id)
          );
        })
      )
    );
  }

  function humanSize(bytes) {
    bytes = Number(bytes || 0);
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + " MB";
    return (bytes / 1024 / 1024 / 1024).toFixed(1) + " GB";
  }

  // ---------------------------------------------------------------------
  // Portfolio page — top-level component
  // ---------------------------------------------------------------------

  function PortfolioPage() {
    const [summary, setSummary] = useState(null);
    const [filters, setFilters] = useState({
      q: "",
      boards: "",
      assignees: "",
      statuses: "",
      include_done: true,
      include_archived: false,
      view: "backlog",
    });
    const [itemsPayload, setItemsPayload] = useState({ items: [], facets: {}, by_status: {}, page: { limit: 0, offset: 0, total: 0 } });
    const [loadingItems, setLoadingItems] = useState(false);
    const [loadError, setLoadError] = useState(null);
    const [selected, setSelected] = useState(null);
    const [selectedDetail, setSelectedDetail] = useState(null);
    const [detailLoading, setDetailLoading] = useState(false);
    const [detailError, setDetailError] = useState(null);
    const [hierarchyOpen, setHierarchyOpen] = useState(true);
    const refreshTimer = useRef(null);

    // Pull summary once on mount.
    useEffect(function () {
      let cancelled = false;
      fetchSummary().then(function (data) {
        if (!cancelled) setSummary(data);
      }).catch(function (err) {
        if (!cancelled) setLoadError(String(err && err.message || err));
      });
      return function () { cancelled = true; };
    }, []);

    // Pull items whenever the filter or view changes.
    useEffect(function () {
      let cancelled = false;
      setLoadingItems(true);
      setLoadError(null);
      const params = {
        q: filters.q,
        boards: filters.boards,
        assignees: filters.assignees,
        statuses: filters.statuses,
        include_done: filters.include_done,
        include_archived: filters.include_archived,
        view: filters.view,
        limit: 500,
        offset: 0,
      };
      fetchItems(params).then(function (data) {
        if (cancelled) return;
        setItemsPayload(data || { items: [], facets: {}, by_status: {}, page: { limit: 0, offset: 0, total: 0 } });
        setLoadingItems(false);
      }).catch(function (err) {
        if (cancelled) return;
        setLoadError(String(err && err.message || err));
        setLoadingItems(false);
      });
      return function () { cancelled = true; };
    }, [
      filters.q, filters.boards, filters.assignees, filters.statuses,
      filters.include_done, filters.include_archived, filters.view,
    ]);

    // Detail fetch when the selected item changes.
    useEffect(function () {
      if (!selected) { setSelectedDetail(null); return; }
      let cancelled = false;
      setDetailLoading(true);
      setDetailError(null);
      fetchItem(selected.board_slug, selected.task_id).then(function (data) {
        if (cancelled) return;
        setSelectedDetail(data);
        setDetailLoading(false);
      }).catch(function (err) {
        if (cancelled) return;
        setDetailError(String(err && err.message || err));
        setDetailLoading(false);
      });
      return function () { cancelled = true; };
    }, [selected && selected.portfolio_id]);

    // Background poll: refresh items + summary every 20s so the
    // dashboard doesn't require a manual reload. Cancellable via
    // cleanup on unmount.
    useEffect(function () {
      function tick() {
        fetchItems({
          q: filters.q,
          boards: filters.boards,
          assignees: filters.assignees,
          statuses: filters.statuses,
          include_done: filters.include_done,
          include_archived: filters.include_archived,
          view: filters.view,
          limit: 500,
          offset: 0,
        }).then(function (data) {
          setItemsPayload(data || { items: [], facets: {}, by_status: {}, page: { limit: 0, offset: 0, total: 0 } });
        }).catch(function () { /* swallow polling errors */ });
        fetchSummary().then(function (data) {
          setSummary(data);
        }).catch(function () { /* ignore */ });
      }
      refreshTimer.current = setInterval(tick, 20000);
      return function () {
        if (refreshTimer.current) clearInterval(refreshTimer.current);
      };
    }, [
      filters.q, filters.boards, filters.assignees, filters.statuses,
      filters.include_done, filters.include_archived, filters.view,
    ]);

    const facets = itemsPayload.facets || {};
    const items = itemsPayload.items || [];
    const byStatus = itemsPayload.by_status || {};
    const total = (itemsPayload.page && itemsPayload.page.total) || items.length;

    function onResetFilters() {
      setFilters({
        q: "",
        boards: "",
        assignees: "",
        statuses: "",
        include_done: true,
        include_archived: false,
        view: filters.view,
      });
    }

    function onSetView(view) {
      setFilters(Object.assign({}, filters, { view: view }));
    }

    return h("div", { className: "hermes-console-root" },
      h("header", { className: "hermes-console-header" },
        h("h1", { className: "hermes-console-title" }, "Portfolio Board"),
        h("p", { className: "hermes-console-subtitle" },
          "Unified, read-only view over canonical Hermes Kanban data. V1.0 — no mutations."
        ),
        h("div", { className: "hermes-console-summary" },
          summary
            ? h(React.Fragment, null,
                h("span", { className: "hermes-console-summary-cell" },
                  h("strong", null, summary.boards || 0), " boards"
                ),
                h("span", { className: "hermes-console-summary-cell" },
                  h("strong", null, summary.items || 0), " items"
                ),
                h("span", { className: "hermes-console-summary-cell" },
                  h("strong", null, summary.active_items || 0), " active"
                ),
                h("span", { className: "hermes-console-summary-cell" },
                  h("strong", null, summary.blocked_items || 0), " blocked"
                ),
                h("span", { className: "hermes-console-summary-cell" },
                  h("strong", null, summary.needs_keith_items || 0), " need Keith"
                )
              )
            : h("span", { className: "hermes-console-muted" }, "Loading summary…")
        )
      ),
      h(FilterBar, {
        filters: filters,
        setFilters: setFilters,
        facets: facets,
        onReset: onResetFilters,
      }),
      loadError
        ? h("div", { className: "hermes-console-error" }, "API error: " + loadError)
        : null,
      h("div", { className: "hermes-console-body" },
        h(HierarchyPane, {
          items: items,
          open: hierarchyOpen,
          onToggle: function () { setHierarchyOpen(!hierarchyOpen); },
          selectedId: selected && selected.portfolio_id,
          onSelect: function (it) { setSelected(it); },
        }),
        h("main", { className: "hermes-console-main" },
          h("div", { className: "hermes-console-main-head" },
            h(Tabs, {
              value: filters.view,
              onValueChange: onSetView,
            },
              h(TabsList, null,
                h(TabsTrigger, { value: "backlog" },
                  "Backlog",
                  h("span", { className: "hermes-console-tab-count" }, total)
                ),
                h(TabsTrigger, { value: "board" },
                  "Board",
                  h("span", { className: "hermes-console-tab-count" }, total)
                )
              )
            ),
            h("div", { className: "hermes-console-main-counts" },
              loadingItems
                ? h("span", { className: "hermes-console-muted" }, "Loading…")
                : h("span", null, total + " items")
            )
          ),
          h(TabsContent, { value: "backlog" },
            h(BacklogTable, {
              items: items,
              onSelect: function (it) { setSelected(it); },
              selectedId: selected && selected.portfolio_id,
            })
          ),
          h(TabsContent, { value: "board" },
            h(BoardView, {
              byStatus: byStatus,
              onSelect: function (it) { setSelected(it); },
              selectedId: selected && selected.portfolio_id,
            })
          )
        ),
        h(DetailDrawer, {
          item: selectedDetail,
          loading: detailLoading,
          error: detailError,
          onClose: function () { setSelected(null); setSelectedDetail(null); },
        })
      )
    );
  }

  // ---------------------------------------------------------------------
  // Register with the dashboard host
  // ---------------------------------------------------------------------

  if (window.__HERMES_PLUGINS__ && typeof window.__HERMES_PLUGINS__.register === "function") {
    window.__HERMES_PLUGINS__.register("hermes-console", PortfolioPage);
  }
})();
