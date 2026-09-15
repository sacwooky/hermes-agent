"""Offline completion and transcript-preservation regression tests.

HTTP and Grok authentication are replaced with synthetic test values. No live
provider, credential file, signer or endpoint is accessed.
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "scripts" / "robin-review-runtime"


def load(name):
    loader = importlib.machinery.SourceFileLoader('completion_' + name, str(RUNTIME / name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


gemini, grok = load('gemini-review'), load('grok-review')


def response(text, finish='stop'):
    return {'id': 'synthetic-response', 'choices': [{'index': 0, 'message': {'content': text}, 'finish_reason': finish}]}


def run_wrapper(module, raw):
    reply = MagicMock()
    reply.__enter__.return_value = reply
    reply.read.return_value = raw if isinstance(raw, bytes) else raw.encode('utf-8')
    stdout, stderr = io.StringIO(), io.StringIO()
    code = 0
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(module.Path, 'cwd', return_value=Path(tmp)))
        stack.enter_context(patch.object(module.urllib.request, 'urlopen', return_value=reply))
        stack.enter_context(patch.object(sys, 'argv', ['review', '-p', 'Synthetic fixture']))
        if module is grok:
            stack.enter_context(patch.object(module, '_key', return_value='synthetic-test-only'))
        stack.enter_context(contextlib.redirect_stdout(stdout))
        stack.enter_context(contextlib.redirect_stderr(stderr))
        try:
            module.main()
        except SystemExit as exc:
            code = exc.code
    return code, stdout.getvalue(), stderr.getvalue()


class CompletionTests(unittest.TestCase):
    def test_long_complete_reviews_keep_prefix_middle_suffix_and_whitespace(self):
        text = '  B1: first finding\r\n' + ('Review evidence. ' * 2000) + '\nM4: final finding\nVERDICT: CHANGES_REQUESTED\n\n'
        for module in (gemini, grok):
            with self.subTest(module=module.__name__):
                code, actual, err = run_wrapper(module, json.dumps(response(text)))
                self.assertEqual((code, actual, err), (0, text, ''))

    def test_pass_shaped_incomplete_response_fails_without_losing_diagnostic_text(self):
        text = 'VERDICT: PASS\nBut required finding details were cut mid-'
        for module in (gemini, grok):
            for finish in ('length', 'content_filter', 'tool_calls', None, 'unknown'):
                with self.subTest(module=module.__name__, finish=finish):
                    code, actual, err = run_wrapper(module, json.dumps(response(text, finish)))
                    self.assertEqual(code, 8)
                    self.assertEqual(actual, text)
                    self.assertIn('did not report normal completion', err)

    def test_missing_completion_signal_cannot_succeed(self):
        data = response('VERDICT: PASS')
        del data['choices'][0]['finish_reason']
        for module in (gemini, grok):
            self.assertNotEqual(run_wrapper(module, json.dumps(data))[0], 0)

    def test_empty_and_nontext_content_cannot_succeed(self):
        for module in (gemini, grok):
            for text in ('', ' \n', None, {'text': 'VERDICT: PASS'}):
                with self.subTest(module=module.__name__, text=text):
                    self.assertNotEqual(run_wrapper(module, json.dumps(response(text)))[0], 0)

    def test_grok_sse_requires_stop_not_just_done(self):
        first = {'id': 'stream1', 'choices': [{'index': 0, 'delta': {'content': 'B1 complete\nVERDICT: CHANGES_REQUESTED'}, 'finish_reason': None}]}
        last = {'id': 'stream1', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
        prefix = 'data: ' + json.dumps(first) + '\n'
        self.assertFalse(grok._extract_response(prefix + 'data: [DONE]\n')[1])
        raw = prefix + 'data: ' + json.dumps(last) + '\ndata: [DONE]\n'
        self.assertEqual(grok._extract_response(raw), ('B1 complete\nVERDICT: CHANGES_REQUESTED', True))

    def test_grok_rejects_mixed_responses_choices_and_text_after_stop(self):
        full = response('VERDICT: PASS')
        another = response('This belongs to another response')
        another['id'] = 'another'
        multi = response('VERDICT: PASS')
        multi['choices'].append({'index': 1, 'message': {'content': 'CHANGES'}, 'finish_reason': 'stop'})
        cases = [json.dumps(full) + json.dumps(another), json.dumps(multi), json.dumps(full) + ' trailing garbage', json.dumps(full) + json.dumps({'error': {'message': 'synthetic'}})]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertFalse(grok._extract_response(raw)[1])

    def test_grok_accepts_terminal_usage_chunk_after_completed_stream(self):
        text = 'B2: detailed finding\nVERDICT: CHANGES_REQUESTED'
        raw = json.dumps(response(text)) + json.dumps({'id': 'synthetic-response', 'choices': [], 'usage': {'completion_tokens': 50}})
        self.assertEqual(grok._extract_response(raw), (text, True))

    def test_grok_never_discards_invalid_content_and_calls_remaining_text_complete(self):
        first = {'id': 'stream1', 'choices': [{'delta': {'content': 'VERDICT: PASS'}, 'finish_reason': None}]}
        bad = {'id': 'stream1', 'choices': [{'delta': {'content': {'finding': 'must not be dropped'}}, 'finish_reason': None}]}
        last = {'id': 'stream1', 'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        code, text, _ = run_wrapper(grok, ''.join(json.dumps(x) for x in (first, bad, last)))
        self.assertEqual((code, text), (8, 'VERDICT: PASS'))

    def test_invalid_utf8_cannot_be_silently_repaired_into_pass(self):
        raw = json.dumps(response('VERDICT: PASS')).encode().replace(b'PASS', b'PA\xffSS')
        for module in (gemini, grok):
            code, text, _ = run_wrapper(module, raw)
            self.assertNotEqual(code, 0)
            self.assertEqual(text, '')

    def test_gemini_rejects_error_envelope_and_conflicting_choices(self):
        bad = response('VERDICT: PASS')
        bad['error'] = {'message': 'synthetic error'}
        multi = response('VERDICT: PASS')
        multi['choices'].append({'index': 1, 'message': {'content': 'VERDICT: CHANGES_REQUESTED'}, 'finish_reason': 'stop'})
        for data in (bad, multi):
            self.assertNotEqual(run_wrapper(gemini, json.dumps(data))[0], 0)

    def test_grok_keeps_earlier_findings_when_later_json_is_cut(self):
        text = 'B1: preserve this entire earlier finding\nVERDICT: CHANGES_REQUESTED'
        raw = json.dumps(response(text, None)) + '{"choices":['
        code, actual, _ = run_wrapper(grok, raw)
        self.assertEqual((code, actual), (8, text))

    def test_sse_keepalive_works_but_done_cannot_be_followed_by_more_review(self):
        first = {'id': 'stream1', 'choices': [{'delta': {'content': 'VERDICT: PASS'}, 'finish_reason': None}]}
        last = {'id': 'stream1', 'choices': [{'delta': {}, 'finish_reason': 'stop'}]}
        prefix = ': keepalive\n\ndata: ' + json.dumps(first) + '\n\n'
        tail = 'data: ' + json.dumps(last) + '\n\n'
        self.assertEqual(grok._extract_response(prefix + ': another keepalive\n' + tail + 'data: [DONE]\n'), ('VERDICT: PASS', True))
        self.assertEqual(grok._extract_response(prefix + 'data: [DONE]\n' + tail), ('VERDICT: PASS', False))


if __name__ == '__main__':
    unittest.main(verbosity=2)
