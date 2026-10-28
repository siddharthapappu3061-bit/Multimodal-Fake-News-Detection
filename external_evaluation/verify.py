"""Regression verification, not external performance measurement."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_external_evaluation import ExternalEvaluationTests
from test_gated_integration import GatedIntegrationTests, OBS
from external_evaluation.evaluate import digest


if __name__ == '__main__':
    protected = [ROOT / name for name in ('six_model_app.py', 'reliability.py', 'gated_integration.py',
        'integrated_inference.py', 'experiments/reliability_combined/policy.json')]
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExternalEvaluationTests)
    for name in ('test_all_six_parity_and_repeatability', 'test_devices_dimensions_caching_and_encoder_reuse',
                 'test_reports_labels_ids_and_promotion_fallback'):
        suite.addTest(GatedIntegrationTests(name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    evidence = {'passed': result.wasSuccessful() and before == after, 'tests_run': result.testsRun,
                'protected_source_and_policy_unchanged': before == after, 'sha256': after,
                'cpu_regression': OBS, 'note': 'Existing Fakeddit fixtures verify regression only; these are not external evaluation results.'}
    (ROOT / 'external_evaluation/verification.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    sys.exit(not evidence['passed'])
