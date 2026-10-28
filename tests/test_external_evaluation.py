"""Synthetic data here tests software only; it is never external evaluation evidence."""
import copy
import csv
import tempfile
import unittest
import json
import numpy as np
from unittest.mock import patch
from pathlib import Path
from PIL import Image
from external_evaluation.evaluate import FIELDS, OPTIONAL, KEYS, validate_dataset, metrics, summarize
from external_evaluation.evaluate import json_text


class ExternalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'images').mkdir()
        Image.new('RGB', (4, 4), 'red').save(self.root / 'images/a.png')
        self.row = dict(zip(FIELDS, ['unit-test-only', 'Synthetic software fixture', 'images\\a.png', 'Fake',
                                  'https://example.org/source', 'https://example.org/evidence', 'Software test only']))

    def write(self, rows):
        path = self.root / 'samples.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS + OPTIONAL)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_relative_windows_path_and_required_fields(self):
        rows, _ = validate_dataset(self.write([self.row]), self.root)
        self.assertTrue(Path(rows[0]['resolved_image_path']).is_file())
        for field, value in [('image_path', 'missing.png'), ('verified_label', '0'),
                             ('verification_notes', ''), ('verification_evidence_url', 'no-url')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_dataset(self.write([{**self.row, field: value}]), self.root)
        with self.assertRaises(ValueError):
            validate_dataset(self.write([]), self.root)

    def test_csv_encodings_preserve_punctuation_and_bytes(self):
        row = {**self.row, 'headline': 'Someone’s “headline” — café'}
        for encoding in ('utf-8', 'utf-8-sig', 'cp1252'):
            with self.subTest(encoding=encoding):
                path = self.write([row])
                text = path.read_text(encoding='utf-8')
                path.write_bytes(text.encode(encoding))
                before = path.read_bytes()
                rows, warnings = validate_dataset(path, self.root)
                self.assertEqual(rows[0]['headline'], row['headline'])
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(bool(warnings), encoding == 'cp1252')
        path.write_bytes(before + bytes([0x81]))
        with self.assertRaisesRegex(ValueError, 'CSV UTF-8'):
            validate_dataset(path, self.root)

    def test_duplicates_and_evidence(self):
        for second in [self.row, {**self.row, 'sample_id': 'another', 'verified_label': 'Real'}]:
            with self.assertRaises(ValueError):
                validate_dataset(self.write([self.row, second]), self.root)
        with self.assertRaises(ValueError):
            validate_dataset(self.write([{**self.row, 'association_status': 'supported_misleading'}]), self.root)
        valid = {**self.row, 'association_status': 'supported_misleading',
                 'association_evidence_url': 'https://example.org/context', 'association_notes': 'Fixture evidence'}
        self.assertEqual(len(validate_dataset(self.write([valid]), self.root)[0]), 1)

    def test_local_fakeddit_overlap(self):
        (self.root / 'data').mkdir()
        (self.root / 'data/train_processed.csv').write_text('id,clean_title\nunit-test-only,something\n')
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            validate_dataset(self.write([self.row]), self.root)

    def test_metrics_fake_positive_and_undefined_auc(self):
        result = metrics([0, 1, 0, 1], [.9, .8, .2, .1])
        self.assertEqual(result['confusion_matrix'], [[1, 1], [1, 1]])
        self.assertEqual(result['fake_f1'], .5)
        self.assertIsNone(metrics([0], [.8])['roc_auc'])
        with self.assertRaises(ValueError):
            metrics([0], [float('nan')])

    def test_display_only_policy_and_error_categories(self):
        rows = [{**self.row, 'verified_label': label} for label in ('Fake', 'Real')]
        outputs = [{'scores': {k: [.99, .01] for k in KEYS}}, {'scores': {k: [.8, .2] for k in KEYS}}]
        before = copy.deepcopy(outputs)
        benchmark = {'models': {k: {'accuracy': .8} for k in KEYS}}
        policy = {'available': True, 'strategy': 'confidence', 'threshold': .95}
        report, records, errors, _ = summarize(rows, outputs, policy, benchmark)
        self.assertEqual(report['gated_selective']['coverage'], .5)
        self.assertEqual(report['gated_selective']['accepted_accuracy'], 1)
        self.assertEqual(report['gated_selective']['abstained_error_rate'], 1)
        self.assertEqual(records[-1]['displayed_decision'], 'Requires Verification')
        self.assertTrue(errors[-1]['false_positive'])
        self.assertFalse(errors[-1]['evidence_supported_misleading_association'])
        self.assertEqual(outputs, before)
        report, records, _, _ = summarize(rows, outputs, {'available': False}, benchmark)
        self.assertEqual(report['gated_selective']['coverage'], 1)
        self.assertEqual(records[-1]['displayed_decision'], 'Fake')

    def test_cli_writes_separate_reports(self):
        # Mock outputs are deliberately synthetic, written only in a temporary test directory.
        import external_evaluation.evaluate as evaluator
        from types import SimpleNamespace
        csv_path = self.write([self.row])
        policy_dir = self.root / 'experiments/reliability_combined'
        policy_dir.mkdir(parents=True)
        (policy_dir / 'policy.json').write_text('{"available":false}')
        (self.root / 'sha256.json').write_text('{}')
        fake = SimpleNamespace(run_ids=('fixture', 'fixture'), gated_run_id='fixture',
            reports={'dashboard': {'models': {k: {'accuracy': .8} for k in KEYS}}},
            analyze=lambda *args: {'scores': {k: [.01, .99] for k in KEYS}})
        with patch.object(evaluator, 'ROOT', self.root), patch.object(evaluator, 'load_policy', return_value={'available': False}), \
             patch('gated_integration.gated_bundle_path', return_value=self.root), \
             patch('gated_integration.SixModelPredictor', return_value=fake), \
             patch('sys.argv', ['evaluate.py', '--csv', str(csv_path)]):
            evaluator.main()
        out = next((self.root / 'external_evaluation/results').iterdir())
        report = json.loads((out / 'metrics.json').read_text())
        self.assertEqual(len(report['models']), 6)
        self.assertEqual(report['gated_selective']['applied_strategy'], 'none')
        self.assertEqual(report['error_counts']['gated_fusion']['overconfident_error'], 1)
        for file in ('raw_outputs.json', 'predictions.csv', 'error_analysis.csv', 'fusion_cases.csv', 'REPORT.md'):
            self.assertTrue((out / file).is_file())

    def test_json_numpy_numbers_are_numeric_and_nonfinite_rejected(self):
        result = json.loads(json_text({'n': np.int64(2), 'p': np.float32(.5),
                                      'flag': np.bool_(True), 'matrix': np.array([[1, 2]])}))
        self.assertEqual(result, {'n': 2, 'p': .5, 'flag': True, 'matrix': [[1, 2]]})
        self.assertIsInstance(result['n'], int)
        for value in (np.float64('nan'), np.float32('inf')):
            with self.assertRaises(ValueError):
                json_text({'bad': value})


if __name__ == '__main__':
    unittest.main()
