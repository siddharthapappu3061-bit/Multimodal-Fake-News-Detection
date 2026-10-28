"""Synthetic fixtures test mechanics only; these are never benchmark results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from gated_model import GatedFusion
from gated_analysis import (metrics, derangement, ablations, choose_threshold, disagreement_score,
                            disagreement_report, warning_policy, gate_statistics, forward_batches)
from clip_experiment import read_cached_modality
import gated_pipeline


class GatedTests(unittest.TestCase):
    def test_shapes_gradients_and_round_trip(self):
        torch.manual_seed(42)
        model = GatedFusion(13, 7)
        text, image = torch.randn(8, 13), torch.randn(8, 7)
        logits, gt, gi = model(text, image)
        self.assertEqual(logits.shape, (8, 2))
        torch.testing.assert_close(gt + gi, torch.ones(8))
        torch.nn.functional.cross_entropy(logits, torch.arange(8) % 2).backward()
        for module in [model.text_branch, model.image_branch, model.gate, model.classifier]:
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)
        with self.assertRaises(ValueError):
            model(text, torch.randn(8, 2048))
        model.eval()
        restored = GatedFusion(**model.spec).eval(); restored.load_state_dict(model.state_dict())
        for actual, expected in zip(restored(text,image), model(text,image)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_thresholds_metrics_and_warning_policy(self):
        text, image = np.array([.9,.1,.8,.2]), np.array([.8,.9,.7,.8])
        self.assertTrue(np.all((disagreement_score(text,image) >= 0) & (disagreement_score(text,image) <= 1)))
        with self.assertRaises(ValueError):
            choose_threshold(text,image,split='test')
        with self.assertRaises(ValueError):
            disagreement_score([2],[.5])
        with self.assertRaises(ValueError):
            gate_statistics([[2,-1]])
        self.assertEqual(metrics([0,1],[.9,.1])['confusion_matrix'], [[1,0],[0,1]])
        threshold = choose_threshold(text,image)
        report = disagreement_report([0,1,0,1], text,image,text,threshold)
        self.assertFalse(warning_policy(report)['changes_label'])
        report['split'] = 'test'
        with self.assertRaises(ValueError):
            warning_policy(report)

    def test_alignment_and_corrupt_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'chunk.pt'
            frame = pd.DataFrame({'id':['a','b'], '2_way_label':[0,1]})
            torch.save(dict(ids=['b','a'],labels=torch.tensor([1,0]),features=torch.tensor([[2.,3.],[4.,5.]])),path)
            torch.testing.assert_close(read_cached_modality(folder,frame,2),torch.tensor([[4.,5.],[2.,3.]]))
            frame.loc[0,'2_way_label'] = 1
            with self.assertRaises(ValueError):
                read_cached_modality(folder,frame,2)

    def test_missing_cache_reports_exact_paths(self):
        root = Path(__file__).resolve().parents[2]
        frames = {s: pd.DataFrame() for s in ['train','validate','test']}
        with tempfile.TemporaryDirectory() as folder, patch.object(gated_pipeline,'fingerprint',return_value='f1dcc535cd73416c8a8c'):
            with self.assertRaises(FileNotFoundError) as caught:
                gated_pipeline.load_caches(frames,root/'windows_demo/artifacts',root/'CLIP-vision/clip_artifacts',Path(folder)/'text',Path(folder)/'clip')
            for split in frames:
                self.assertIn(str(Path(folder)/'text/f1dcc535cd73416c8a8c'/split/'text'/'*.pt'),str(caught.exception))
                self.assertIn(str(Path(folder)/'clip/93baa53bd7fff1bd4c8d'/split/'*.pt'),str(caught.exception))

    def test_ablation_uses_training_mean_and_no_self_swaps(self):
        for n in (2,3,20):
            mapping = derangement(n,np.random.default_rng(42))
            self.assertTrue(np.all(mapping != np.arange(n)))
            self.assertEqual(sorted(mapping), list(range(n)))
        class RecordingHead(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.inputs = []
            def forward(self,text,image):
                self.inputs.append((text.clone(),image.clone()))
                score = text[:,0] + image[:,0]
                return torch.stack([score,-score],1)
        head = RecordingHead()
        data = dict(text=torch.randn(8,3),image=torch.randn(8,2),labels=torch.arange(8)%2,ids=[str(i) for i in range(8)])
        training = dict(text=torch.ones(12,3)*4,image=torch.ones(12,2)*7)
        with tempfile.TemporaryDirectory() as folder:
            report = ablations(head,data,training,folder,repetitions=2)
            self.assertTrue(any(torch.equal(t,torch.ones(8,3)*4) for t,i in head.inputs))
            self.assertTrue(any(torch.equal(i,torch.ones(8,2)*7) for t,i in head.inputs))
            self.assertIn('out of distribution',report['limits'])
            self.assertTrue((Path(folder)/'ablation.png').is_file())

    def test_recovery_matches_uninterrupted_training(self):
        # Tiny CPU fixtures exercise exact optimizer/RNG recovery, never Fakeddit training.
        torch.set_num_threads(2)
        torch.manual_seed(1)
        data = {split:dict(text=torch.randn(10,5),image=torch.randn(10,3),labels=torch.arange(10)%2) for split in ['train','validate']}
        config = dict(lr=.0003,batch_size=5,epochs=3,patience=5,weight_decay=.001)
        with tempfile.TemporaryDirectory() as folder, patch.object(gated_pipeline,'gpu',return_value='cpu'):
            full = gated_pipeline.train_seed(data,Path(folder)/'full',config,42,'fixture')
            real_save = gated_pipeline.atomic_save
            def interrupted_save(value,path):
                real_save(value,path)
                if Path(path).name == 'recovery.pt' and value['epoch'] == 0:
                    raise RuntimeError('Simulated interruption after epoch saved')
            with patch.object(gated_pipeline,'atomic_save',side_effect=interrupted_save):
                with self.assertRaisesRegex(RuntimeError,'Simulated'):
                    gated_pipeline.train_seed(data,Path(folder)/'resume',config,42,'fixture')
            recovered = gated_pipeline.train_seed(data,Path(folder)/'resume',config,42,'fixture')
            for name, tensor in full['state_dict'].items():
                torch.testing.assert_close(tensor,recovered['state_dict'][name],rtol=0,atol=0)

    def test_notebook_code_and_embedded_sources(self):
        from build_notebook import build, HERE
        notebook = build()
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                code = ''.join(line for line in cell['source'] if not line.startswith('%'))
                compile(code, f'cell-{i}', 'exec')
        import ast
        source_cell = next(c for c in notebook['cells'] if 'SOURCE_FILES = ' in ''.join(c['source']))
        tree = ast.parse(''.join(source_cell['source']))
        sources = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_FILES' for t in n.targets))
        for name, source in sources.items():
            compile(source,name,'exec')
            if name.startswith('gated_'):
                self.assertEqual(source,(HERE/name).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
