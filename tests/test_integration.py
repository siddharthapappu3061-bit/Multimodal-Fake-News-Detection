"""Real-weight Windows CPU integration checks. No training or dataset writes."""
import json
import os
import sys
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
from PIL import Image
import torch
from streamlit.testing.v1 import AppTest

from integration_support import bundle_paths, asset_signature, load_evaluation, MODEL_NAMES, DEFAULT_MODEL, read_json
from integration_runtime import load_predictor

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
BASELINE, CLIP = bundle_paths(ROOT)
OBSERVATIONS = {}


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.signature = asset_signature(BASELINE, CLIP)
        cls.predictor = load_predictor(str(BASELINE), str(CLIP), cls.signature)
        cls.samples = pd.read_csv(ROOT / "data/test_processed.csv", dtype={"id": str}).set_index("id")
        cls.exported = pd.read_csv(CLIP / "results/test_predictions.csv", dtype={"id": str}).set_index("id")
        OBSERVATIONS["baseline_run_id"], OBSERVATIONS["clip_run_id"] = cls.predictor.run_ids

    def sample(self, sample_id):
        from dataset import resolve_image
        row = self.samples.loc[sample_id]
        with Image.open(resolve_image(ROOT / "data", row.local_image_path)) as image:
            return row.clean_title, image.copy()

    def test_actual_checkpoint_shapes_devices_and_freezing(self):
        p = self.predictor
        self.assertEqual(len(p.baseline.heads) + len(p.clip_heads), 5)
        models = [p.baseline.encoders.text, p.baseline.encoders.image, p.clip.model, *p.baseline.heads.values(), *p.clip_heads.values()]
        for model in models:
            self.assertFalse(model.training)
            self.assertTrue(all(param.device.type == "cpu" and not param.requires_grad for param in model.parameters()))
        title, image = self.sample("bdlmf3")
        t, r, c = p.baseline.encoders.encode_text([title]), p.baseline.encoders.encode_images([image]), p.clip.encode([image])
        self.assertEqual(tuple(t.shape), (1, 768))
        self.assertEqual(tuple(r.shape), (1, 2048))
        self.assertEqual(tuple(c.shape), (1, 512))
        for kind, head in {**p.baseline.heads, **p.clip_heads}.items():
            self.assertEqual(tuple(head(t, c if kind.startswith("clip_") else r).shape), (1, 2))

    def test_predictions_repeat_and_match_exported_colab(self):
        from integration_support import REPORT_KEYS
        observations = {}
        for sample_id in ("bdlmf3", "ba8vze"):
            title, image = self.sample(sample_id)
            first = self.predictor.predict(title, image)
            second = self.predictor.predict(title, image)
            self.assertEqual(set(first), set(MODEL_NAMES))
            self.assertEqual(first, second)
            for kind, probs in first.items():
                self.assertEqual(len(probs), 2)
                self.assertAlmostEqual(sum(probs), 1.0, places=6)
                expected = float(self.exported.loc[sample_id, REPORT_KEYS[kind] + "_fake_probability"])
                torch.testing.assert_close(torch.tensor(probs[0]), torch.tensor(expected), atol=2e-3, rtol=2e-3)
            # Original Predictor.predict is still called unchanged and gives the same results.
            original = self.predictor.baseline.predict(title, image)
            for kind, probs in original.items():
                self.assertEqual(first[kind], probs)
            observations[sample_id] = first
        OBSERVATIONS["samples"] = observations

    def test_clip_export_cpu_smoke_values_match(self):
        smoke = read_json(CLIP / "results/cpu_smoke.json")
        title, image = self.sample(smoke["sample_id"])
        actual = self.predictor.predict(title, image)
        for kind, expected in smoke["probabilities"].items():
            torch.testing.assert_close(torch.tensor(actual[kind]), torch.tensor(expected), atol=2e-3, rtol=2e-3)

    def test_encoders_cached_and_text_computed_once(self):
        again = load_predictor(str(BASELINE), str(CLIP), self.signature)
        self.assertIs(again, self.predictor)
        title, image = self.sample("ba8vze")
        with patch.object(again.baseline.encoders, "encode_text", wraps=again.baseline.encoders.encode_text) as text:
            again.predict(title, image)
            self.assertEqual(text.call_count, 1)
        # The CLIP branch has no second text encoder.
        self.assertFalse(hasattr(again, "text"))

    def test_missing_artifacts_report_exact_paths(self):
        missing = ROOT / "nonexistent_clip_bundle_for_test"
        with self.assertRaises(FileNotFoundError) as caught:
            asset_signature(BASELINE, missing)
        self.assertIn(str(missing / "models/clip_fusion.pt"), str(caught.exception))
        self.assertIn(str(missing / "encoders/clip/model.safetensors"), str(caught.exception))

    def test_verified_dashboard_metrics(self):
        report = load_evaluation(BASELINE, CLIP)
        self.assertEqual(report["count"], 500)
        expected = {"text": .810, "image": .696, "fusion": .822, "clip_image": .756, "clip_fusion": .834}
        for kind, accuracy in expected.items():
            self.assertAlmostEqual(report["models"][kind]["accuracy"], accuracy)
            self.assertIn("macro_f1", report["models"][kind])

    def test_ui_default_and_all_pages(self):
        app = AppTest.from_file(str(ROOT / "integrated_app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="primary_model").value, DEFAULT_MODEL)
        for page in ("Model Comparison", "Dataset Analysis", "Model Architecture"):
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            if page == "Model Comparison":
                self.assertEqual(len(app.dataframe[0].value), 5)
        app.sidebar.radio[0].set_value("Fake News Detection").run()
        app.button[0].click().run()
        self.assertTrue(app.warning)

    def test_ui_analyze_uses_real_weights_and_preserves_results_on_selection(self):
        # AppTest lacks a file upload setter; supply actual image bytes at the widget boundary.
        title, image = self.sample("ba8vze")
        stream = BytesIO()
        image.save(stream, format="PNG")
        with patch("streamlit.file_uploader", return_value=stream):
            app = AppTest.from_file(str(ROOT / "integrated_app.py"), default_timeout=60).run()
            app.text_area(key="headline").set_value(title).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.metric), 6)  # Primary selection + all five comparisons.
            self.assertEqual(set(app.session_state["analysis"]["scores"]), set(MODEL_NAMES))
            with patch.object(self.predictor, "predict", side_effect=AssertionError("Selection should reuse predictions")):
                app.selectbox(key="primary_model").set_value("text").run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IntegrationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    OBSERVATIONS.update({"passed": result.wasSuccessful(), "tests_run": result.testsRun,
                         "comparison_tolerance": {"atol": .002, "rtol": .002},
                         "device": "cpu", "training_performed": False})
    target = ROOT / "results/integration_verification.json"
    target.write_text(json.dumps(OBSERVATIONS, indent=2), encoding="utf-8")
    raise SystemExit(not result.wasSuccessful())
