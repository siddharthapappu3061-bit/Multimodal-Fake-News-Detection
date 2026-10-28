import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import pandas as pd

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights

import pipeline
from dataset import resolve_image, extract_zip
from models import Head, masked_mean, ENCODER_CONFIG, HEAD_CONFIG
from inference import Predictor

ROOT = Path(__file__).resolve().parents[1]


class PrototypeTests(unittest.TestCase):
    def test_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = Path(tmp).resolve() / 'images/train/a.jpg'
            for raw in ('data\\images\\train\\a.jpg', 'C:\\old\\project\\data\\images\\train\\a.jpg', '/old/data/images/train/a.jpg'):
                self.assertEqual(resolve_image(tmp, raw), expected)
            with self.assertRaises(ValueError):
                resolve_image(tmp, 'images/../../outside.jpg')

    def test_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'data.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('project/data/train_processed.csv', 'id\na\n')
            self.assertEqual(extract_zip(archive, Path(tmp) / 'out'), Path(tmp) / 'out/project/data')
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../escape.txt', 'bad')
            with self.assertRaises(ValueError):
                extract_zip(archive, Path(tmp) / 'bad')

    def test_pool_and_image_transform(self):
        pooled = masked_mean(torch.tensor([[[2., 4.], [9., 9.]]]), torch.tensor([[1, 0]]))
        torch.testing.assert_close(pooled, torch.tensor([[2., 4.]]))
        image = ResNet50_Weights.IMAGENET1K_V2.transforms()(Image.new('RGB', (300, 200)))
        self.assertEqual(tuple(image.shape), (3, 224, 224))

    def test_heads_cpu_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('text', 'image', 'fusion'):
                model = Head(kind).eval()
                t, i = torch.randn(2, 768), torch.randn(2, 2048)
                pred = model(t, i)
                self.assertEqual(tuple(pred.shape), (2, 2))
                path = Path(tmp) / 'head.pt'
                pipeline.atomic_save(model.state_dict(), path)
                restored = Head(kind).eval()
                restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
                torch.testing.assert_close(pred, restored(t, i))

    def test_training_recovery(self):
        # Small synthetic fixtures test mechanics only; they are never reported as dataset metrics.
        torch.set_num_threads(2)
        features = {s: {'text': torch.randn(8, 768), 'image': torch.randn(8, 2048),
                        'labels': torch.tensor([0, 1] * 4), 'ids': list(map(str, range(8)))}
                    for s in ('train', 'validate')}
        config = dict(seed=42, epochs=2, batch_size=4, lr=.001, weight_decay=.0001, patience=5)
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'gpu', return_value=torch.device('cpu')):
            reference, resumed = Path(tmp) / 'ref', Path(tmp) / 'resumed'
            pipeline.train('fusion', features, reference, config, 'test')
            original_save = pipeline.atomic_save

            def interrupt(obj, path):
                original_save(obj, path)
                raise InterruptedError('simulated disconnect after checkpoint')

            with patch.object(pipeline, 'atomic_save', side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    pipeline.train('fusion', features, resumed, config, 'test')
            pipeline.train('fusion', features, resumed, config, 'test')
            a = torch.load(reference / 'models/fusion.pt', weights_only=True)
            b = torch.load(resumed / 'models/fusion.pt', weights_only=True)
            for name in a['state_dict']:
                torch.testing.assert_close(a['state_dict'][name], b['state_dict'][name], rtol=0, atol=0)

    def test_feature_cache_resume(self):
        class CountingEncoders:
            calls = 0

            def encode_text(self, titles):
                self.calls += 1
                return torch.ones(len(titles), 768)

            def encode_images(self, images):
                self.calls += 1
                return torch.ones(len(images), 2048)

        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'gpu', return_value=torch.device('cpu')):
            image = Path(tmp) / 'image.png'
            Image.new('RGB', (224, 224)).save(image)
            df = pd.DataFrame({'id': ['a', 'b'], 'clean_title': ['one', 'two'], '2_way_label': [0, 1], 'resolved_path': [str(image)] * 2})
            encoder = CountingEncoders()
            cache = Path(tmp) / 'cache'
            first = pipeline.extract({'test': df}, encoder, cache, 1)
            self.assertEqual(encoder.calls, 4)
            second = pipeline.extract({'test': df}, encoder, cache, 1)
            self.assertEqual(encoder.calls, 4)
            torch.testing.assert_close(first['test']['text'], second['test']['text'])
            (cache / 'test/image/000001-0001.pt').unlink()
            pipeline.extract({'test': df}, encoder, cache, 1)
            self.assertEqual(encoder.calls, 5)

    def test_inference_contract(self):
        class StubEncoders:
            def __init__(self, *args, **kwargs):
                pass

            def encode_text(self, titles):
                return torch.ones(len(titles), 768)

            def encode_images(self, images):
                return torch.ones(len(images), 2048)

        with tempfile.TemporaryDirectory() as tmp, patch('inference.Encoders', StubEncoders):
            bundle = Path(tmp)
            pipeline.write_json({'encoders': ENCODER_CONFIG, 'heads': HEAD_CONFIG, 'labels': {'0': 'Fake', '1': 'Real'}, 'run_id': 'test'}, bundle / 'config.json')
            for kind in ('text', 'image', 'fusion'):
                pipeline.atomic_save({'kind': kind, 'run_id': 'test', 'state_dict': Head(kind).state_dict()}, bundle / 'models' / f'{kind}.pt')
            predictor = Predictor(bundle)
            for probabilities in predictor.predict('A headline', Image.new('RGB', (224, 224))).values():
                self.assertAlmostEqual(sum(probabilities), 1., places=6)

    def test_notebook_sources(self):
        nb = json.loads((ROOT / 'colab_training.ipynb').read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                code = ''.join(line for line in cell['source'] if not line.startswith('%'))
                ast.parse(code)
        source_cell = ''.join(nb['cells'][4]['source'])
        tree = ast.parse(source_cell)
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCE_FILES' for t in n.targets))
        for name, embedded in ast.literal_eval(assignment.value).items():
            source = 'baseline_app.py' if name == 'app.py' else name
            self.assertEqual(embedded, (ROOT / source).read_text(encoding='utf-8'))

    def test_evaluation_outputs(self):
        data = {'test': {'text': torch.randn(500, 768), 'image': torch.randn(500, 2048),
                         'labels': torch.tensor([0, 1] * 250), 'ids': [str(i) for i in range(500)]}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'gpu', return_value=torch.device('cpu')), patch.dict(os.environ, {'MPLCONFIGDIR': str(ROOT / '.mpl-cache')}):
            run = Path(tmp)
            for kind in ('text', 'image', 'fusion'):
                pipeline.atomic_save({'state_dict': Head(kind).state_dict(), 'run_id': 'fixture'}, run / 'models' / f'{kind}.pt')
                pipeline.write_json([{'epoch': 1, 'train_loss': 1., 'val_loss': 1.}], run / 'results' / f'{kind}_history.json')
            result = pipeline.evaluate(data, run, 'fixture')
            self.assertEqual(result['test_count'], 500)
            self.assertEqual(len(pd.read_csv(run / 'results/predictions.csv')), 500)
            for metrics in result['models'].values():
                self.assertEqual(sum(map(sum, metrics['confusion_matrix'])), 500)
                self.assertTrue(0 <= metrics['roc_auc'] <= 1)


if __name__ == '__main__':
    unittest.main()
