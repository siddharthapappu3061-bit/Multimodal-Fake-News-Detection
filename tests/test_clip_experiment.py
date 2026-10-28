"""Small offline mechanics tests. No pretrained downloads or benchmark training."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast

import clip_experiment as experiment
from clip_models import ClipHead, ClipEncoder, FEATURE_SPEC
from clip_inference import ClipPredictor
from models import Head
from pipeline import atomic_save, write_json

ROOT = Path(__file__).resolve().parents[1]


class ClipExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dimensions_are_configurable_and_reject_resnet_input(self):
        for width in (512, 32):
            for kind in experiment.KINDS:
                model = ClipHead(kind, text_dim=16, image_dim=width).eval()
                self.assertEqual(model(torch.ones(3, 16), torch.ones(3, width)).shape, (3, 2))
                restored = ClipHead(**model.spec).eval()
                restored.load_state_dict(model.state_dict())
                torch.testing.assert_close(model(torch.ones(3, 16), torch.ones(3, width)), restored(torch.ones(3, 16), torch.ones(3, width)))
        with self.assertRaisesRegex(ValueError, 'dimension'):
            ClipHead('clip_image')(None, torch.ones(1, 2048))

    def test_cache_alignment_and_label_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = pd.DataFrame({'id': ['first', 'second'], '2_way_label': [0, 1]})
            path = Path(tmp) / 'out_of_order.pt'
            part = {'ids': ['second', 'first'], 'labels': torch.tensor([1, 0]), 'features': torch.tensor([[2., 2.], [1., 1.]])}
            atomic_save(part, path)
            actual = experiment.read_cached_modality(tmp, frame, 2)
            torch.testing.assert_close(actual, torch.tensor([[1., 1.], [2., 2.]]))
            part['labels'] = torch.tensor([0, 1])
            atomic_save(part, path)
            with self.assertRaisesRegex(ValueError, 'labels disagree'):
                experiment.read_cached_modality(tmp, frame, 2)
            part['labels'] = torch.tensor([1, 0])
            part['ids'] = ['extra', 'first']
            atomic_save(part, path)
            with self.assertRaisesRegex(ValueError, 'Missing/extra'):
                experiment.read_cached_modality(tmp, frame, 2)

    def test_protected_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for output in (root / 'baseline', root / 'baseline/experiments', root):
                with self.assertRaises(ValueError):
                    experiment.guard_output(output, root / 'baseline')
            experiment.guard_output(root / 'experiment', root / 'baseline')

    def test_fixed_shards_resume_across_batch_size_changes_and_corruption(self):
        class FakeEncoder:
            dimension = 512
            calls = 0

            def encode(self, images):
                self.calls += 1
                return torch.ones(len(images), 512)

        with tempfile.TemporaryDirectory() as tmp, patch.object(experiment, 'gpu', return_value=torch.device('cpu')):
            image = Path(tmp) / 'image.jpg'
            Image.new('RGB', (20, 20)).save(image)
            frame = pd.DataFrame({'id': [str(i) for i in range(130)], '2_way_label': [0, 1] * 65, 'resolved_path': [str(image)] * 130})
            encoder = FakeEncoder()
            cache = Path(tmp) / 'cache'
            first = experiment.extract_clip({'train': frame}, encoder, cache, 'data', batch_size=64)
            self.assertEqual(encoder.calls, 3)
            second = experiment.extract_clip({'train': frame}, encoder, cache, 'data', batch_size=16)
            self.assertEqual(encoder.calls, 3)
            torch.testing.assert_close(first['train'], second['train'])
            (cache / 'data/train/000128.pt').write_bytes(b'incomplete')
            experiment.extract_clip({'train': frame}, encoder, cache, 'data', batch_size=16)
            self.assertEqual(encoder.calls, 4)

    def test_training_recovery_matches_uninterrupted_and_never_reads_test(self):
        data = {split: {'text': torch.randn(8, 16), 'image': torch.randn(8, 32), 'labels': torch.tensor([0, 1] * 4)}
                for split in ('train', 'validate')}
        data['test'] = object()  # Any attempt to use test data in training would fail.
        cfg = dict(seed=42, epochs=3, batch_size=4, lr=.001, weight_decay=.0001, patience=5, dropout=.3)
        with tempfile.TemporaryDirectory() as tmp, patch.object(experiment, 'gpu', return_value=torch.device('cpu')):
            root = Path(tmp)
            ref = experiment.train_trial('clip_fusion', data, root / 'reference', cfg, 'fixture')
            save = experiment.atomic_save

            def interrupt(value, path):
                save(value, path)
                raise InterruptedError('Simulated disconnect after persisted epoch')

            with patch.object(experiment, 'atomic_save', side_effect=interrupt), self.assertRaises(InterruptedError):
                experiment.train_trial('clip_fusion', data, root / 'resume', cfg, 'fixture')
            recovered = experiment.train_trial('clip_fusion', data, root / 'resume', cfg, 'fixture')
            a, b = [torch.load(p, weights_only=True) for p in (ref, recovered)]
            self.assertEqual(a['validation_macro_f1'], b['validation_macro_f1'])
            for key in a['state_dict']:
                torch.testing.assert_close(a['state_dict'][key], b['state_dict'][key], rtol=0, atol=0)
            # A finished trial must not take another optimizer step.
            with patch('torch.optim.AdamW.step', side_effect=AssertionError('Retraining a completed trial')):
                experiment.train_trial('clip_fusion', data, root / 'resume', cfg, 'fixture')

    def test_metrics_and_validation_only_gate(self):
        metrics = experiment.binary_metrics([0, 0, 1, 1], [[.8, .2], [.4, .6], [.1, .9], [.2, .8]])
        self.assertEqual(metrics['confusion_matrix'], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics['precision'], 1.)
        self.assertAlmostEqual(metrics['recall'], .5)
        self.assertAlmostEqual(metrics['f1'], 2 / 3)
        self.assertAlmostEqual(metrics['macro_f1'], ((2 / 3) + .8) / 2)
        validation = {'split': 'validate', 'models': {name: {'macro_f1': score} for name, score in
            [('resnet50', .6), ('distilbert_resnet50', .8), ('clip_image', .7), ('clip_fusion', .8)]}}
        gate = experiment.promotion_gate(validation)
        self.assertTrue(gate['candidates']['clip_image']['eligible'])
        self.assertFalse(gate['candidates']['clip_fusion']['eligible'])
        with self.assertRaises(ValueError):
            experiment.promotion_gate({**validation, 'split': 'test'})

    def test_five_model_comparison_and_baseline_files_unchanged(self):
        # Synthetic data only; all outputs disappear with the temporary directory.
        ids = [str(i) for i in range(500)]
        text, labels = torch.randn(500, 768), torch.tensor([0, 1] * 250)
        baseline = {'validate': {'ids': ids, 'labels': labels, 'text': text, 'image': torch.randn(500, 2048)}}
        clip = {'validate': {'ids': ids, 'labels': labels, 'text': text, 'image': torch.randn(500, 512)}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(experiment, 'gpu', return_value=torch.device('cpu')):
            root = Path(tmp)
            original, run = root / 'original', root / 'experiment'
            write_json({'run_id': 'old'}, original / 'config.json')
            for kind in ('text', 'image', 'fusion'):
                atomic_save({'kind': kind, 'run_id': 'old', 'state_dict': Head(kind).state_dict()}, original / 'models' / f'{kind}.pt')
            for kind in experiment.KINDS:
                head = ClipHead(kind)
                atomic_save({'run_id': 'new', 'spec': head.spec, 'state_dict': head.state_dict()}, run / 'models' / f'{kind}.pt')
            before = {p.name: experiment.sha_file(p) for p in (original / 'models').glob('*.pt')}
            results = experiment.compare_models(baseline, clip, original, run, 'new')
            self.assertEqual(len(results['validate']['models']), 5)
            for metric in results['validate']['models'].values():
                self.assertEqual(sum(map(sum, metric['confusion_matrix'])), 500)
                self.assertIn('macro_f1', metric)
            after = {p.name: experiment.sha_file(p) for p in (original / 'models').glob('*.pt')}
            self.assertEqual(before, after)
            with patch.dict(os.environ, {'MPLCONFIGDIR': str(ROOT / '.mpl-cache')}):
                experiment.plot_comparison(run, 'validate')
            self.assertTrue((run / 'results/validate_comparison.png').is_file())

    def test_real_tiny_encoders_offline_cpu_export_loading_and_freezing(self):
        # Tiny random transformers exercise the actual APIs/serialization, not model quality.
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            clip_assets, text_assets = bundle / 'encoders/clip', bundle / 'encoders/distilbert'
            vision = CLIPVisionModelWithProjection(CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                num_attention_heads=4, projection_dim=512, image_size=32, patch_size=16))
            vision.save_pretrained(clip_assets)
            CLIPImageProcessor(size={'shortest_edge': 32}, crop_size={'height': 32, 'width': 32}).save_pretrained(clip_assets)
            encoder = ClipEncoder(clip_assets)
            self.assertFalse(encoder.model.training)
            self.assertTrue(all(not p.requires_grad for p in encoder.model.parameters()))
            encoded = encoder.encode([Image.new('RGB', (40, 40))])
            self.assertEqual(encoded.shape, (1, 512))
            torch.testing.assert_close(encoded.norm(dim=1), torch.ones(1))
            text_assets.mkdir(parents=True)
            (text_assets / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n', encoding='utf-8')
            tokenizer = DistilBertTokenizerFast(vocab_file=str(text_assets / 'vocab.txt'))
            tokenizer.save_pretrained(text_assets)
            text = DistilBertModel(DistilBertConfig(vocab_size=len(tokenizer), dim=16, hidden_dim=32, n_layers=1, n_heads=2, max_position_embeddings=128))
            text.save_pretrained(text_assets)
            config = {'run_id': 'tiny', 'feature_spec': FEATURE_SPEC, 'labels': {'0': 'Fake', '1': 'Real'}, 'heads': {}}
            for kind in experiment.KINDS:
                head = ClipHead(kind, text_dim=16)
                config['heads'][kind] = head.spec
                atomic_save({'run_id': 'tiny', 'spec': head.spec, 'state_dict': head.state_dict()}, bundle / 'models' / f'{kind}.pt')
            write_json(config, bundle / 'config.json')
            predictor = ClipPredictor(bundle)
            probabilities = predictor.predict('hello world', Image.new('RGB', (40, 40)))
            self.assertEqual(set(probabilities), set(experiment.KINDS))
            self.assertTrue(all(not p.requires_grad for p in predictor.text.model.parameters()))
            for scores in probabilities.values():
                self.assertAlmostEqual(sum(scores), 1., places=6)

    def test_notebook_syntax_and_embedded_sources(self):
        notebook = json.loads((ROOT / 'colab_clip_experiment.ipynb').read_text(encoding='utf-8'))
        source_files = None
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(line for line in cell['source'] if not line.startswith('%'))
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCE_FILES' for t in node.targets):
                    source_files = ast.literal_eval(node.value)
        self.assertIsNotNone(source_files)
        self.assertNotIn('app.py', source_files)
        for name, source in source_files.items():
            self.assertEqual(source, (ROOT / name).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
