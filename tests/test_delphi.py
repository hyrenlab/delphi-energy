import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import delphi
from examples.offline import BRIEF, INTAKE, ScriptedLLM


class ConfigTests(unittest.TestCase):
    def test_budget_rejects_zero_negative_nan_infinity_and_non_numbers(self):
        for value in [0, -1, 'nan', 'inf', '-inf', 'oops', None, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                delphi._valid_amount(value, 'budget', positive=True)

    def test_price_config_validation_and_zero_rates(self):
        self.assertEqual(delphi._load_prices('{"local/model": [0, 0]}'), {'local/model': (0, 0)})
        for raw in ['no JSON', '[]', '{"model": [1, 2]}', '{"/model": [1,2]}',
                    '{"p/m": [1]}', '{"p/m": [-1, 2]}', '{"p/m": [1, NaN]}',
                    '{"p/m": [true, 2]}', '{"p/m": [1, "inf"]}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                delphi._load_prices(raw)

    def test_configured_price_and_explicit_fallback(self):
        with patch.multiple(delphi, MODEL_PRICES={'fixture/model': (2, 8)}, FALLBACK_PRICES=(1, 5)):
            self.assertEqual(delphi._estimate_cost('fixture', 'model', 1_000_000, 500_000), 6)
            self.assertEqual(delphi._estimate_cost('unknown', 'model', 1_000_000, 500_000), 3.5)
            self.assertIn('hypothetical fallback 1/5', delphi._pricing_note())

    def test_budget_boundary_warns_at_80_and_skips_at_100(self):
        with patch.object(delphi, 'BUDGET_USD_LIMIT', 1):
            for amount, allowed, warnings in [(0.79, True, 0), (0.8, True, 1), (0.99, True, 1), (1, False, 1), (1.1, False, 1)]:
                with self.subTest(amount=amount):
                    transcript = delphi.DelphiTranscript('fixture')
                    transcript.emit('fixture', cost_usd=amount)
                    self.assertEqual(delphi._budget_check_or_warn(transcript), allowed)
                    self.assertEqual(sum(e.type == 'budget_warning' for e in transcript.events), warnings)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.directory = Path(directory)
        self.stack.enter_context(patch.multiple(delphi, LEDGER_DIR=self.directory,
            _STREAM_ENABLED=False, BUDGET_USD_LIMIT=1, MODEL_PRICES={}, FALLBACK_PRICES=(1, 5)))
        self.stack.enter_context(patch.dict(delphi.ADAPTERS, {
            'memory': delphi.NoOpMemoryAdapter(), 'bias': delphi.NoOpBiasAdapter(),
            'notifier': delphi.NoOpNotifierAdapter()}))

    def run_script(self, responses):
        llm = ScriptedLLM(responses)
        with patch.dict(delphi.ADAPTERS, {'llm': llm}):
            transcript = delphi.run_delphi('synthetic fixture')
        self.assertEqual(len(llm.calls), len(responses))
        self.assertFalse(llm.responses)
        return transcript, llm

    def test_lite_path_is_five_sequential_calls(self):
        transcript, llm = self.run_script([json.dumps(INTAKE), 'advocate', 'skeptic', 'realist', json.dumps(BRIEF)])
        self.assertEqual([e.payload['role'] for e in transcript.events if e.type == 'role_opening'],
                         [delphi._ROLE_META[k][0] for k in ['advocate', 'skeptic', 'realist']])
        self.assertEqual(transcript.brief, BRIEF)
        self.assertEqual(len([e for e in transcript.events if e.type == 'action']), 1)
        self.assertEqual(list(self.directory.iterdir()), [])
        self.assertIn('Estimated cost', transcript.render())

    def test_standard_round_two_is_supplementary_and_visible(self):
        intake = dict(INTAKE, reversibility='low', cost_of_being_wrong='medium')
        brief = dict(BRIEF, pre_mortem='An unexpected venue closure leaves attendees without a meeting place and destroys trust in the format.')
        responses = [json.dumps(intake)] + ['opening'] * 7 + ['round one', 'red team', 'alternatives', 'evidence audit', json.dumps(brief), 'round two challenge']
        transcript, llm = self.run_script(responses)
        stages = [e.type for e in transcript.events if e.cost_usd is not None]
        self.assertEqual(stages, ['intake'] + ['role_opening'] * 7 + ['cross_exam', 'red_team', 'counterfactual', 'evidence_audit', 'judge', 'cross_exam'])
        self.assertIn('evidence audit', llm.calls[12]['messages'][1]['content'])
        self.assertEqual(transcript.brief, brief)
        self.assertIn('have not been revised', transcript.render_brief_only())
        self.assertIn('does not revise', transcript.render())
        action = next(e for e in transcript.events if e.type == 'action')
        self.assertEqual(action.payload['first_domino'], brief['first_domino'])

    def test_high_stakes_includes_dissent_and_counts_grouped_calls(self):
        intake = dict(INTAKE, reversibility='low', cost_of_being_wrong='high')
        dissent = {'agreement_level': 'partial', 'should_surface': True,
                   'divergence_summary': 'fixture disagreement'}
        responses = [json.dumps(intake)] + ['opening'] * 7 + [
            'round one', 'red team', 'alternate skeptic', json.dumps(dissent),
            'alternatives', 'evidence audit', json.dumps(BRIEF)]
        transcript, llm = self.run_script(responses)
        self.assertEqual(transcript.llm_call_count(), 15)
        self.assertEqual(transcript.total_tokens(), (1500, 750))
        self.assertIn('fixture disagreement', llm.calls[-1]['messages'][1]['content'])
        self.assertIn('15 LLM calls', transcript.render())

    def test_exhausted_soft_budget_still_runs_judge(self):
        with patch.object(delphi, 'BUDGET_USD_LIMIT', 0.0001):
            transcript, _ = self.run_script([json.dumps(INTAKE), json.dumps(BRIEF)])
        self.assertGreater(transcript.total_cost(), 0.0001)
        self.assertFalse(any(e.type == 'role_opening' for e in transcript.events))
        self.assertEqual(transcript.brief, BRIEF)

    def test_judge_json_repair_succeeds_and_tracks_both_calls(self):
        transcript, _ = self.run_script([json.dumps(INTAKE), 'a', 's', 'r', 'broken JSON', json.dumps(BRIEF)])
        self.assertEqual(transcript.brief, BRIEF)
        repair = next(e for e in transcript.events if e.type == 'judge_repair')
        self.assertTrue(repair.payload['ok'])
        self.assertEqual(transcript.total_tokens(), (600, 300))

    def test_judge_json_failure_keeps_raw_text_without_action(self):
        transcript, _ = self.run_script([json.dumps(INTAKE), 'a', 's', 'r', 'not JSON', 'still not JSON'])
        self.assertIsNone(transcript.brief)
        self.assertFalse(any(e.type == 'action' for e in transcript.events))
        self.assertIn('not JSON', transcript.render())

    def test_json_parser_rejects_non_object_and_handles_fences(self):
        for value in ['', 'broken', '[]', '[{"ok": true}]', 'null', '42', '"text"']:
            self.assertIsNone(delphi._safe_json_load(value))
        self.assertEqual(delphi._safe_json_load('```json\n{"ok": true}\n```'), {'ok': True})

    def test_dry_run_never_calls_adapters_or_writes_ledger(self):
        with patch.dict(delphi.ADAPTERS, {k: None for k in delphi.ADAPTERS}):
            transcript = delphi.run_delphi('preview', dry_run=True)
        self.assertEqual(transcript.total_cost(), 0)
        self.assertIn('[DRY-RUN]', transcript.render())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_zero_cost_calls_still_count_in_transcript(self):
        with patch.object(delphi, 'FALLBACK_PRICES', (0, 0)):
            transcript, _ = self.run_script([json.dumps(INTAKE), 'a', 's', 'r', json.dumps(BRIEF)])
        self.assertIn('5 LLM calls', transcript.render())


if __name__ == '__main__':
    unittest.main()
