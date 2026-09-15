"""Zero-inference retention regressions using real runner and SQLite recorder.

Run: scripts/run_tests.sh tests/hermes_cli/test_robin_review_retention.py -- -q
All provider processes are mocked. All keys, homes, manifests and DBs are synthetic
temporary fixtures; no configured credentials or live services are accessed.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "scripts" / "robin-review-runtime"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "robin_review_runtime"
SOURCE = ROOT


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def structured(verdict="approved"):
    return {"verdict": verdict, "confidence": 0.9, "scorecard": {"fixture": 9},
            "blocking_issues": [], "advisory_issues": [], "missing_skill_findings": [],
            "required_repair_actions": [], "evidence_reviewed": ["synthetic fixture"],
            "calibration_substrate_flags": []}


class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (SOURCE / "hermes_cli" / "kanban_autonomy.py").is_file():
            raise RuntimeError("The repository checkout must contain hermes_cli/kanban_autonomy.py")
        cleanup = contextlib.ExitStack()
        cls.addClassCleanup(cleanup.close)
        cls.import_temp = cleanup.enter_context(tempfile.TemporaryDirectory(prefix="retention-import-"))
        temporary = Path(cls.import_temp)
        old_path = list(sys.path)
        cleanup.callback(lambda: sys.path.__setitem__(slice(None), old_path))
        module_name = "hermes_cli.review_loop.sc_review"
        missing = object()
        old_module = sys.modules.get(module_name, missing)
        old_parent = sys.modules.get("hermes_cli.review_loop")
        old_attribute = getattr(old_parent, "sc_review", missing)

        def restore_module():
            if old_module is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module
            parent = sys.modules.get("hermes_cli.review_loop")
            if parent is not None:
                if old_attribute is missing:
                    parent.__dict__.pop("sc_review", None)
                else:
                    parent.sc_review = old_attribute

        cleanup.callback(restore_module)
        sys.path.insert(0, str(SOURCE))
        with contextlib.ExitStack() as import_scope:
            import_scope.enter_context(patch.dict(os.environ, {
                "HERMES_HOME": str(temporary), "HERMES_KANBAN_HOME": str(temporary)}, clear=True))
            import_scope.enter_context(patch.object(Path, "home", return_value=temporary))
            from hermes_cli import kanban_autonomy, kanban_db
            cls.ka, cls.kb = kanban_autonomy, kanban_db
            cls.scr = load_source(module_name, ROOT / "hermes_cli" / "review_loop" / "sc_review.py")
            import hermes_cli.review_loop
            hermes_cli.review_loop.sc_review = cls.scr

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="retention-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.packet = self.root / "packet"
        self.packet.mkdir()
        (self.packet / "review-request.md").write_text("author-provider: openai\nSynthetic acceptance criteria.\n", encoding="utf-8")
        (self.packet / "diff.patch").write_text("synthetic diff fixture\n", encoding="utf-8")
        env = {"HERMES_HOME": str(self.root), "HERMES_KANBAN_HOME": str(self.root),
               "HERMES_KANBAN_DB": str(self.root / "kanban.db")}
        self.enterContext(patch.dict(os.environ, env, clear=True))
        self.enterContext(patch.object(Path, "home", return_value=self.root))
        self.key = self.root / "synthetic-review-key"
        self.key.write_text("01" * 32, encoding="ascii")
        fake = types.ModuleType("fusionkit")
        fake.authorship = types.SimpleNamespace(
            AuthorDeclarationError=ValueError,
            parse_author=lambda _: ("openai", "synthetic-explicit-declaration"),
            assert_panel_independent=lambda author, models: None)
        fake.routing = types.SimpleNamespace(provider_of=lambda m: (
            "google" if "gemini" in m else "anthropic" if "claude" in m else "xai"))
        self.enterContext(patch.dict(sys.modules, {"fusionkit": fake}))
        self.enterContext(patch.dict(sys.modules))
        self.runner = load_source("synthetic_retention_runner", RUNTIME / "review-deterministic.py")
        self.runner.VERDICT_DIR = self.root / "verdicts"
        self.runner.VERDICT_KEY_PATH = self.key
        self.enterContext(patch.object(self.ka, "DEFAULT_VERDICT_KEY_PATH", str(self.key)))
        self.enterContext(patch.object(self.runner, "_lane_env", return_value={}))
        self.enterContext(patch.object(self.runner.shutil, "which", return_value="synthetic-review-binary"))
        self.process = self.enterContext(patch.object(self.runner.subprocess, "run", side_effect=AssertionError("Unexpected process; no live calls allowed")))

    def create_task(self):
        self.kb.init_db()
        with contextlib.closing(self.kb.connect()) as conn:
            task = self.kb.create_task(conn, title="Synthetic review retention", assignee="builder")
            self.kb.claim_task(conn, task)
        return task

    def configure_outputs(self, outputs):
        positions = {}
        def run(cmd, **kwargs):
            self.assertIs(kwargs["text"], False)
            model = cmd[cmd.index("--model") + 1]
            lane = next(lane for lane, chain in self.runner.LANE_MODEL_CHAINS.items() if model in chain)
            values = outputs[lane]
            if not isinstance(values, list):
                values = [values]
            position = positions.get(lane, 0)
            positions[lane] = position + 1
            value = values[min(position, len(values) - 1)]
            if isinstance(value, BaseException):
                raise value
            if isinstance(value, tuple):
                code, body = value
            else:
                code, body = 0, value
            return types.SimpleNamespace(returncode=code, stdout=body, stderr=b"private diagnostic sentinel")
        self.process.side_effect = run

    def invoke_main(self, task):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["review-deterministic.py", task, "a" * 40, str(self.packet), "HIGH"]), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.runner.main()
        return json.loads(stdout.getvalue()), stderr.getvalue()

    def signed(self, status):
        return json.loads(Path(status["verdict_file"]).read_text(encoding="utf-8"))

    def assert_binding(self, payload):
        binding = payload["provider_output_manifest"]
        manifest = self.runner.VERDICT_DIR / binding["manifest_path"]
        raw = manifest.read_bytes()
        self.assertEqual(binding["manifest_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(binding["manifest_bytes"], len(raw))
        data = json.loads(raw)
        self.assertEqual(binding["attempts"], data["attempts"])
        self.assertEqual(data["task_id"], payload["task_id"])
        for entry in data["attempts"]:
            received = (manifest.parent / entry["stdout_path"]).read_bytes()
            self.assertEqual(entry["stdout_sha256"], hashlib.sha256(received).hexdigest())
            self.assertEqual(entry["stdout_bytes"], len(received))
        return manifest, data

    def test_complete_signed_reviews_and_real_sqlite_repair_handoff(self):
        task = self.create_task()
        outputs = {lane: (f"  BEGIN-{lane} café\r\n" + "α detail " * 1600 + f"\r\nMIDDLE-{lane}\r\n" +
                          "substantive explanation " * 800 + f"\r\nEND-{lane}\r\nVERDICT: CHANGES_REQUESTED\r\n  ").encode("utf-8")
                   for lane in ("gemini", "claude", "grok")}
        self.configure_outputs(outputs)
        status, stderr = self.invoke_main(task)
        signed = self.signed(status)
        payload = signed["payload"]
        self.assertTrue(status["verdict_saved"])
        self.assertTrue(status["kanban_recorded"])
        self.assertEqual(payload["verdict"], "CHANGES_REQUESTED")
        self.assertNotIn("private diagnostic sentinel", stderr)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.assertTrue(self.ka.verify_verdict_signature(canonical, signed["signature"], key_path=str(self.key)))
        manifest, attempts = self.assert_binding(payload)
        for review, attempt in zip(payload["provider_reviews"], attempts["attempts"]):
            expected = outputs[review["lane"]]
            self.assertEqual(review["review_text"].encode("utf-8"), expected)
            self.assertEqual((manifest.parent / attempt["stdout_path"]).read_bytes(), expected)
            self.assertIn(review["review_text"], payload["evidence"])
            self.assertEqual(review["review_sha256"], hashlib.sha256(expected).hexdigest())
            self.assertEqual(review["review_utf8_bytes"], len(expected))
        self.assertEqual(payload["findings"]["reviews"], payload["provider_reviews"])
        with contextlib.closing(self.kb.connect()) as conn:
            self.assertEqual(self.kb.get_task(conn, task).status, "ready")
            comments = self.kb.list_comments(conn, task)
            comment = next(c.body for c in comments if "REVIEW BLOCK" in c.body)
            retained = json.loads(next(line[2:] for line in comment.splitlines() if line.startswith("- {")))
            self.assertEqual(retained, payload["findings"])
            event = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='verdict_recorded'", (task,)).fetchone()
            self.assertIsNone(json.loads(event[0])["findings_count"])

    def test_tampered_early_finding_fails_actual_signature_and_recorder(self):
        task = self.create_task()
        self.configure_outputs({lane: b"BEGIN blocking detail\n" + b"long detail " * 2000 + b"\nVERDICT: CHANGES_REQUESTED\n" for lane in ("gemini", "claude", "grok")})
        status, _ = self.invoke_main(task)
        signed = self.signed(status)
        tampered = copy.deepcopy(signed["payload"])
        tampered["provider_reviews"][0]["review_text"] = tampered["provider_reviews"][0]["review_text"].replace("BEGIN", "FORGED", 1)
        wire = json.dumps(tampered, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.assertFalse(self.ka.verify_verdict_signature(wire.encode("utf-8"), signed["signature"], key_path=str(self.key)))
        with contextlib.closing(self.kb.connect()) as conn:
            result = self.ka.record_review_verdict(conn, task, wire, signed["signature"], fetched_via="robin-ssh", key_path=str(self.key))
            self.assertFalse(result["ok"])
            self.assertIn("HMAC", result["reason"])
            self.assertNotEqual(self.kb.get_task(conn, task).status, "done")

    def test_nonzero_legacy_pass_never_accepted_and_exact_diagnostics_retained(self):
        self.configure_outputs({"gemini": (7, b"VERDICT: PASS\r\npartial\r\n")})
        store = self.runner.ReviewEvidenceStore("task", "a" * 40)
        out, model, err = self.runner.run_lane("gemini", "fixture", str(self.packet), "diff.patch", capture=store.capture)
        self.assertIsNone(out)
        self.assertIsNone(model)
        self.assertNotIn("private diagnostic", err)
        self.assertEqual(len(store.entries), len(self.runner.LANE_MODEL_CHAINS["gemini"]))
        for entry in store.entries:
            self.assertFalse(entry["accepted_review"])
            self.assertEqual(entry["returncode"], 7)
            self.assertEqual((store.directory / entry["stdout_path"]).read_bytes(), b"VERDICT: PASS\r\npartial\r\n")

    def test_nonzero_structured_pass_is_not_parsed_or_accepted(self):
        self.configure_outputs({"gemini": (8, json.dumps(structured()).encode())})
        store = self.runner.ReviewEvidenceStore("task", "commit")
        parse = Mock(side_effect=AssertionError("Nonzero output must not be parsed"))
        result = self.runner.run_structured_lane("gemini", "fixture", str(self.packet), parse, capture=store.capture)
        self.assertIsNone(result[0])
        parse.assert_not_called()
        self.assertTrue(all(not entry["accepted_review"] for entry in store.entries))

    def test_single_sc_nonzero_valid_pass_aborts_without_signature(self):
        self.configure_outputs({"gemini": (9, json.dumps(structured()).encode())})
        with patch.object(self.runner, "_sign_and_record") as sign, contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as raised:
                self.runner.run_sc_review("task", "commit", self.packet, "fixture", "prd")
        self.assertEqual(raised.exception.code, 3)
        sign.assert_not_called()
        self.assertIn("received output:", stderr.getvalue())
        attempts = list(self.runner.VERDICT_DIR.glob("provider-reviews-*/attempt-*.stdout"))
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].read_bytes(), json.dumps(structured()).encode())

    def test_structured_parser_exception_retains_raw_before_aborting(self):
        raw = b'{"untrusted":"parser-trigger"}'
        self.configure_outputs({"gemini": raw})
        store = self.runner.ReviewEvidenceStore("task", "commit")
        with self.assertRaises(RecursionError):
            self.runner.run_structured_lane("gemini", "fixture", str(self.packet), Mock(side_effect=RecursionError), capture=store.capture)
        self.assertEqual(self.process.call_count, 1)
        self.assertEqual((store.directory / "attempt-001.stdout").read_bytes(), raw)
        self.assertFalse(store.entries[0]["accepted_review"])

    def test_single_sc_parser_exception_retains_raw_without_signature(self):
        raw = b'{"untrusted":"parser-trigger"}'
        self.configure_outputs({"gemini": raw})
        with patch.object(self.scr, "parse_sc_reply", side_effect=RecursionError), patch.object(self.runner, "_sign_and_record") as sign:
            with self.assertRaises(RecursionError):
                self.runner.run_sc_review("task", "commit", self.packet, "fixture", "general")
        sign.assert_not_called()
        attempt = next(self.runner.VERDICT_DIR.glob("provider-reviews-*/attempt-001.stdout"))
        self.assertEqual(attempt.read_bytes(), raw)

    def test_timeout_partial_survives_successful_same_provider_fallback(self):
        partial = b"early finding\r\npartial \xff bytes"
        self.configure_outputs({"gemini": [subprocess.TimeoutExpired(["synthetic"], 900, output=partial), b"  full review\r\nVERDICT: FAIL\r\n "]})
        store = self.runner.ReviewEvidenceStore("task", "commit")
        out, model, _ = self.runner.run_lane("gemini", "fixture", str(self.packet), "diff.patch", capture=store.capture)
        self.assertEqual(model, self.runner.LANE_MODEL_CHAINS["gemini"][1])
        self.assertEqual(out, "  full review\r\nVERDICT: FAIL\r\n ")
        self.assertTrue(store.entries[0]["timed_out"])
        self.assertFalse(store.entries[0]["accepted_review"])
        self.assertTrue(store.entries[1]["accepted_review"])
        self.assertEqual((store.directory / "attempt-001.stdout").read_bytes(), partial)

    def test_single_sc_timeout_retains_partial_and_aborts(self):
        self.configure_outputs({"gemini": subprocess.TimeoutExpired(["synthetic"], 900, output=b"partial SC finding\r\n")})
        with patch.object(self.runner, "_sign_and_record") as sign, contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit):
                self.runner.run_sc_review("task", "commit", self.packet, "fixture", "prd")
        sign.assert_not_called()
        self.assertIn("received output:", stderr.getvalue())
        self.assertEqual(next(self.runner.VERDICT_DIR.glob("provider-reviews-*/attempt-001.stdout")).read_bytes(), b"partial SC finding\r\n")

    def test_total_outage_has_no_signed_verdict_and_keeps_all_attempts(self):
        self.configure_outputs({lane: (2, b"partial output\r\n") for lane in ("gemini", "claude", "grok")})
        with patch.object(self.runner, "_sign_and_record") as sign, patch.object(sys, "argv", ["runner", "task", "commit", str(self.packet), "HIGH"]), contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as raised:
                self.runner.main()
        self.assertEqual(raised.exception.code, 3)
        sign.assert_not_called()
        self.assertIn("received output:", stderr.getvalue())
        self.assertEqual(list(self.runner.VERDICT_DIR.glob("*.json")), [])
        self.assertEqual(len(list(self.runner.VERDICT_DIR.glob("provider-reviews-*/attempt-*.stdout"))), 7)

    def test_code_panel_all_nonzero_valid_passes_cannot_sign(self):
        self.configure_outputs({lane: (2, json.dumps(structured()).encode()) for lane in ("gemini", "claude", "grok")})
        with patch.object(self.runner, "_sign_and_record") as sign, contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as raised:
                self.runner.run_code_panel_review("task", "commit", self.packet, "fixture", "diff.patch")
        self.assertEqual(raised.exception.code, 3)
        sign.assert_not_called()
        self.assertIn("received output:", stderr.getvalue())
        self.assertEqual(len(list(self.runner.VERDICT_DIR.glob("provider-reviews-*/attempt-*.stdout"))), 7)

    def test_single_sc_complete_stdout_is_signed_and_bound_without_stripping(self):
        raw = ("  \r\n" + json.dumps(structured("rejected_for_revision")) + "\r\n  ").encode()
        self.configure_outputs({"gemini": raw})
        with patch.object(self.kb, "connect", side_effect=RuntimeError("synthetic unavailable board")), contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()):
            self.runner.run_sc_review("task", "commit", self.packet, "fixture", "prd")
        payload = self.signed(json.loads(stdout.getvalue()))["payload"]
        self.assertEqual(payload["provider_reviews"][0]["review_text"].encode(), raw)
        self.assertEqual(payload["review_type"], "prd")
        manifest, _ = self.assert_binding(payload)
        self.assertEqual((manifest.parent / "attempt-001.stdout").read_bytes(), raw)

    def test_single_sc_hollow_parsed_object_is_retained_but_not_marked_accepted(self):
        self.configure_outputs({"gemini": b"{}"})
        with patch.object(self.runner, "_sign_and_record", return_value=("rejected (hollow)", self.root / "mock-verdict.json", "synthetic-key-id")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.runner.run_sc_review("task", "commit", self.packet, "fixture", "prd")
        manifest = next(self.runner.VERDICT_DIR.glob("provider-reviews-*/manifest-001.json"))
        self.assertFalse(json.loads(manifest.read_bytes())["attempts"][0]["accepted_review"])
        self.assertEqual((manifest.parent / "attempt-001.stdout").read_bytes(), b"{}")

    def test_complete_earlier_lane_survives_later_timeout_and_degraded_policy_unchanged(self):
        task = self.create_task()
        self.configure_outputs({"gemini": b"first complete\r\nVERDICT: PASS\r\n", "claude": subprocess.TimeoutExpired(["synthetic"], 900, output=b"partial claude"), "grok": (1, b"partial grok")})
        status, _ = self.invoke_main(task)
        payload = self.signed(status)["payload"]
        self.assertEqual(payload["verdict"], "PASS")
        self.assertEqual(payload["providers_reviewed"], ["gemini"])
        self.assertEqual(payload["providers_unavailable"], ["claude", "grok"])
        self.assertEqual(payload["findings"], [])
        self.assertEqual(payload["provider_reviews"][0]["review_text"], "first complete\r\nVERDICT: PASS\r\n")
        self.assert_binding(payload)

    def test_capture_failure_aborts_instead_of_skipping_provider(self):
        self.configure_outputs({"gemini": b"VERDICT: PASS"})
        capture = Mock(side_effect=self.runner.ReviewPersistenceError("synthetic disk failure"))
        with self.assertRaises(self.runner.ReviewPersistenceError):
            self.runner.run_lane("gemini", "fixture", str(self.packet), "diff.patch", capture=capture)
        self.assertEqual(self.process.call_count, 1)

    def test_whole_panel_capture_failure_stops_before_next_provider_or_signature(self):
        self.configure_outputs({lane: b"VERDICT: PASS" for lane in ("gemini", "claude", "grok")})
        atomic = self.runner._atomic_private_write
        def fail_first_capture(path, data):
            if Path(path).name == "attempt-001.stdout":
                raise self.runner.ReviewPersistenceError("synthetic full disk")
            return atomic(path, data)
        with patch.object(self.runner, "_atomic_private_write", side_effect=fail_first_capture), patch.object(self.runner, "_sign_and_record") as sign, patch.object(sys, "argv", ["runner", "task", "commit", str(self.packet), "HIGH"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(self.runner.ReviewPersistenceError):
                self.runner.main()
        sign.assert_not_called()
        self.assertEqual(self.process.call_count, 1)

    def test_manifest_failure_preserves_earlier_manifest_and_raw_attempt(self):
        store = self.runner.ReviewEvidenceStore("task", "commit")
        store.capture("gemini", "synthetic-model", b"early full review", returncode=0, accepted=True)
        initial = store.manifest
        atomic = self.runner._atomic_private_write
        def fail_next_manifest(path, data):
            if Path(path).name == "manifest-002.json":
                raise self.runner.ReviewPersistenceError("synthetic publication failure")
            return atomic(path, data)
        with patch.object(self.runner, "_atomic_private_write", side_effect=fail_next_manifest):
            with self.assertRaises(self.runner.ReviewPersistenceError):
                store.capture("claude", "synthetic-model", b"later complete review", returncode=0, accepted=True)
        self.assertEqual(json.loads(initial.read_bytes())["attempts"][0]["stdout_sha256"], hashlib.sha256(b"early full review").hexdigest())
        self.assertEqual((store.directory / "attempt-002.stdout").read_bytes(), b"later complete review")

    def test_atomic_publication_never_replaces_existing_evidence(self):
        destination = self.root / "existing.json"
        destination.write_bytes(b"original complete artifact")
        with self.assertRaises(self.runner.ReviewPersistenceError):
            self.runner._atomic_private_write(destination, b"replacement")
        self.assertEqual(destination.read_bytes(), b"original complete artifact")
        self.assertEqual(list(self.root.glob(".review-write-*")), [])

    def test_write_fsync_and_publish_failures_do_not_publish_partial_file(self):
        for operation in ("mkstemp", "fsync", "link"):
            with self.subTest(operation=operation):
                destination = self.root / (operation + ".json")
                target = self.runner.tempfile if operation == "mkstemp" else self.runner.os
                with patch.object(target, operation, side_effect=OSError("synthetic I/O failure")):
                    with self.assertRaises(self.runner.ReviewPersistenceError):
                        self.runner._atomic_private_write(destination, b"complete bytes")
                self.assertFalse(destination.exists())
                self.assertEqual(list(self.root.glob(".review-write-*")), [])

    def test_partial_write_failure_removes_only_temporary_file(self):
        destination = self.root / "never-published.json"
        original_fdopen = self.runner.os.fdopen
        @contextlib.contextmanager
        def interrupted_writer(fd, mode):
            with original_fdopen(fd, mode) as handle:
                def interrupted_write(data):
                    handle.write(data[:5])
                    raise OSError("synthetic disk full after partial write")
                yield types.SimpleNamespace(write=interrupted_write)
        with patch.object(self.runner.os, "fdopen", side_effect=interrupted_writer):
            with self.assertRaises(self.runner.ReviewPersistenceError):
                self.runner._atomic_private_write(destination, b"full intended evidence")
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".review-write-*")), [])

    def test_transcript_directory_sync_failure_prevents_review_call(self):
        with patch.object(self.runner, "_sync_directory", side_effect=OSError("synthetic directory sync failure")), patch.object(sys, "argv", ["runner", "task", "commit", str(self.packet), "HIGH"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(self.runner.ReviewPersistenceError):
                self.runner.main()
        self.process.assert_not_called()

    @unittest.skipIf(os.name == "nt", "POSIX mode/directory-fsync proof requires Linux; Windows does not provide POSIX ACL proof")
    def test_private_modes_and_directory_fsync_after_publication(self):
        fsync = self.runner.os.fsync
        with patch.object(self.runner.os, "fsync", wraps=fsync) as sync:
            store = self.runner.ReviewEvidenceStore("task", "commit")
            store.capture("gemini", "model", b"review", returncode=0)
        self.assertGreaterEqual(sync.call_count, 6)
        self.assertEqual(stat.S_IMODE(store.directory.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((store.directory / "attempt-001.stdout").stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(store.manifest.stat().st_mode), 0o600)

    def test_signed_artifact_failure_precedes_any_board_call(self):
        payload = {"task_id": "task", "commit": "commit", "recorded_at": 123, "verdict": "FAIL", "lane": "fixture"}
        with patch.object(self.runner, "_atomic_private_write", side_effect=self.runner.ReviewPersistenceError("synthetic disk full")), patch.object(self.kb, "connect") as connect:
            with self.assertRaises(self.runner.ReviewPersistenceError):
                self.runner._sign_and_record("task", payload)
        connect.assert_not_called()

    def test_signed_artifact_is_complete_before_real_board_recorder_runs(self):
        task = self.create_task()
        payload = {"task_id": task, "commit": "commit", "recorded_at": 123, "verdict": "FAIL", "lane": "fixture", "findings": ["synthetic blocker"]}
        actual = self.ka.record_review_verdict
        def inspect_then_record(*args, **kwargs):
            saved = list(self.runner.VERDICT_DIR.glob("*.json"))
            self.assertEqual(len(saved), 1)
            artifact = json.loads(saved[0].read_bytes())
            self.assertEqual(artifact["payload"], payload)
            return actual(*args, **kwargs)
        with patch.object(self.ka, "record_review_verdict", side_effect=inspect_then_record):
            status, artifact, _ = self.runner._sign_and_record(task, payload)
        self.assertEqual(status, "recorded (block)")
        self.assertTrue(artifact.is_file())

    def test_board_failure_keeps_signed_verdict_and_explicit_false_writeback_status(self):
        self.configure_outputs({lane: b"complete review\nVERDICT: FAIL\n" for lane in ("gemini", "claude", "grok")})
        with patch.object(self.kb, "connect", side_effect=RuntimeError("private database diagnostic")):
            status, stderr = self.invoke_main("task")
        self.assertTrue(status["verdict_saved"])
        self.assertFalse(status["kanban_recorded"])
        self.assertEqual(status["kanban"], "error")
        self.assertNotIn("private database diagnostic", stderr)
        self.assertEqual(self.signed(status)["payload"]["verdict"], "FAIL")

    def test_route_models_quorum_and_input_limits_unchanged(self):
        import ast
        def constants(path):
            result = {}
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in {"LANE_TIMEOUT", "LANES", "LANE_MODEL_CHAINS", "ALLOWED", "SC_TYPES"}:
                            result[target.id] = ast.literal_eval(node.value)
            return result
        self.assertEqual(constants(RUNTIME / "review-deterministic.py"), constants(FIXTURES / "before" / "review-deterministic.py"))

    def test_actual_before_evidence_assignment_loses_early_findings_and_candidate_keeps_them(self):
        import ast
        raw = "BEGIN concrete blocking issue\n" + "full review detail " * 1600 + "\nEND verdict\nVERDICT: FAIL\n"
        def evaluate_actual_assignment(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
            evidence = next(node for node in main.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "evidence" for target in node.targets))
            namespace = {"LANE_LABEL": {"gemini": "Gemini"}, "models_used": {"gemini": "fixture"}, "reviews": {"gemini": raw}, "PANEL": ["gemini"]}
            exec(compile(ast.Module(body=[evidence], type_ignores=[]), str(path), "exec"), namespace)
            return namespace["evidence"]
        self.assertNotIn("BEGIN concrete blocking issue", evaluate_actual_assignment(FIXTURES / "before" / "review-deterministic.py"))
        self.assertIn(raw, evaluate_actual_assignment(RUNTIME / "review-deterministic.py"))



class RetentionImportIsolationTests(unittest.TestCase):
    def check_import_cleanup(self, fail_import, initially_present):
        import hermes_cli.review_loop as parent
        name = "hermes_cli.review_loop.sc_review"
        missing = object()
        old_module = sys.modules.get(name, missing)
        old_attribute = getattr(parent, "sc_review", missing)
        module_sentinel = types.ModuleType("original_sc_module")
        attribute_sentinel = object()
        replacement = types.ModuleType("replacement_sc_module")
        original_environment = dict(os.environ)
        original_home_method = Path.__dict__["home"]
        original_path = list(sys.path)
        temporary_paths = []

        class Fixture(unittest.TestCase):
            setUpClass = classmethod(RetentionTests.setUpClass.__func__)

        def fake_load(module_name, path):
            temporary_paths.append(Path(os.environ["HERMES_HOME"]))
            self.assertEqual(Path.home(), temporary_paths[-1])
            self.assertTrue(temporary_paths[-1].is_dir())
            sys.modules[module_name] = replacement
            parent.sc_review = replacement
            if fail_import:
                raise RuntimeError("synthetic setup import failure")
            return replacement

        try:
            if initially_present:
                sys.modules[name] = module_sentinel
                parent.sc_review = attribute_sentinel
            else:
                sys.modules.pop(name, None)
                parent.__dict__.pop("sc_review", None)
            with patch.object(sys.modules[__name__], "load_source", side_effect=fake_load):
                if fail_import:
                    with self.assertRaisesRegex(RuntimeError, "synthetic setup import failure"):
                        Fixture.setUpClass()
                else:
                    Fixture.setUpClass()
            # Environment and Path.home must be restored even before class cleanup.
            self.assertEqual(dict(os.environ), original_environment)
            self.assertIs(Path.__dict__["home"], original_home_method)
            Fixture.doClassCleanups()
            self.assertEqual(Fixture.tearDown_exceptions, [])
            self.assertEqual(sys.path, original_path)
            self.assertEqual(len(temporary_paths), 1)
            self.assertFalse(temporary_paths[0].exists())
            if initially_present:
                self.assertIs(sys.modules[name], module_sentinel)
                self.assertIs(parent.sc_review, attribute_sentinel)
            else:
                self.assertNotIn(name, sys.modules)
                self.assertFalse(hasattr(parent, "sc_review"))
        finally:
            Fixture.doClassCleanups()
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
            if old_attribute is missing:
                parent.__dict__.pop("sc_review", None)
            else:
                parent.sc_review = old_attribute

    def test_failed_setup_restores_import_state(self):
        for initially_present in (False, True):
            with self.subTest(initially_present=initially_present):
                self.check_import_cleanup(fail_import=True, initially_present=initially_present)

    def test_successful_setup_restores_import_state(self):
        for initially_present in (False, True):
            with self.subTest(initially_present=initially_present):
                self.check_import_cleanup(fail_import=False, initially_present=initially_present)

if __name__ == "__main__":
    unittest.main()
