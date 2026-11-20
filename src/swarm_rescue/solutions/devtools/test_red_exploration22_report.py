"""Necessary report semantics: invalid full scores and retained missing data."""
import json
from pathlib import Path
import tempfile
import unittest

from swarm_rescue.solutions.devtools.red_exploration22_report import (
    _raw_phase, load_assets, score_row, summarize,
)


class ReportValidityTests(unittest.TestCase):
    def setUp(self):
        self.item = dict(name='Map06', round=1, seed=2026111901, controller='candidate_rl22')
        self.red = dict(status='ok', exit_code=0, submitted=True, valid_shape=True,
                        phase1=dict(score_map=75., score_wall=55., score_time=20.))

    def test_valid_red_invalid_blue_keeps_raw_total_only(self):
        blue = dict(status='ok', exit_code=0, walltime_limited=True,
                    phase1=dict(score_status='final', round_score=69.5, score_bomb=10.))
        row = score_row(self.item, self.red, blue, [], ['walltime_limit'])
        self.assertTrue(row['red_valid'])
        self.assertFalse(row['blue_valid'])
        self.assertFalse(row['full_valid'])
        self.assertEqual(row['raw_total'], 69.5)
        self.assertIsNone(row['effective_total'])
        self.assertEqual(row['timeouts'], ['blue'])

    def test_unreceipted_attempt_and_missing_assets_are_retained(self):
        with tempfile.TemporaryDirectory(prefix='rl22-report-') as temporary:
            phase = Path(temporary) / 'red'
            attempt = phase / 'attempt_001'
            attempt.mkdir(parents=True)
            raw = dict(self.red, status='process_timeout')
            (attempt / 'result.json').write_text(json.dumps(raw), encoding='utf-8')
            result, selected, issues, attempts = _raw_phase(phase, 'unused')
            self.assertEqual(result, raw)
            self.assertEqual(selected, attempt)
            self.assertIn('missing_verified_done', issues)
            self.assertEqual(attempts, [str(attempt)])
            row = score_row(self.item, result, {}, issues, ['not_run'])
            row['red_attempt'] = str(attempt)
            assets = load_assets(row)
            self.assertFalse(row['red_valid'])
            self.assertEqual(row['map_score'], 75.)
            self.assertTrue(row['artifact_issues'])
            self.assertEqual(assets, {})

    def test_smoke_rejected_before_formal_atlas(self):
        with tempfile.TemporaryDirectory(prefix='rl22-report-') as temporary:
            folder = Path(temporary)
            (folder / 'plan.json').write_text(json.dumps(dict(configuration=dict(smoke=True))), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SMOKE INPUT'):
                summarize(folder)


if __name__ == '__main__':
    unittest.main()
