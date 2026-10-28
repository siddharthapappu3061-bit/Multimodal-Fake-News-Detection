"""Independent Windows CPU inference. Instantiate once and reuse across predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from clip_models import ClipEncoder, CachedTextEncoder, ClipHead, FEATURE_SPEC
from models import Head
from gated_model import GatedFusion


class GatedPredictor:
    def __init__(self, bundle):
        bundle = Path(bundle)
        self.config = json.loads((bundle / 'config.json').read_text())
        if self.config['labels'] != {'0': 'Fake', '1': 'Real'} or self.config['feature_spec'] != FEATURE_SPEC:
            raise ValueError('Incompatible label mapping or preprocessing')
        manifest = json.loads((bundle / 'sha256.json').read_text())
        for name, expected in manifest.items():
            path = (bundle / name).resolve()
            if not path.is_relative_to(bundle.resolve()) or not path.is_file():
                raise FileNotFoundError(f'Missing or unsafe bundle artifact: {name}')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Bundle checksum mismatch: {name}')
        saved = torch.load(bundle / 'gated_fusion.pt', map_location='cpu', weights_only=True)
        if saved['run_id'] != self.config['run_id'] or saved['spec'] != self.config['spec']:
            raise ValueError('Gated checkpoint/configuration mismatch')
        self.gated = GatedFusion(**saved['spec']).eval().requires_grad_(False)
        self.gated.load_state_dict(saved['state_dict'])
        self.text = CachedTextEncoder(bundle / 'encoders/distilbert', device='cpu')
        self.image = ClipEncoder(bundle / 'encoders/clip', device='cpu', local=True)
        self.heads = {}
        for name in ('text', 'clip_image', 'clip_fusion'):
            state = torch.load(bundle / 'models' / (name + '.pt'), map_location='cpu', weights_only=True)
            model = Head('text') if name == 'text' else ClipHead(**state['spec'])
            model.load_state_dict(state['state_dict'])
            self.heads[name] = model.eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(self, title, image):
        if not title.strip():
            raise ValueError('Text must be nonempty')
        text, visual = self.text.encode([title.strip()]), self.image.encode([image])
        logits, gt, gi = self.gated(text, visual)
        probabilities = {name: model(text, visual).softmax(1)[0].tolist() for name, model in self.heads.items()}
        probabilities['gated_fusion'] = logits.softmax(1)[0].tolist()
        disagreement = abs(probabilities['text'][0] - probabilities['clip_image'][0])
        policy = self.config['warning_policy']
        return {'probabilities_order': ['Fake', 'Real'], 'probabilities': probabilities,
                'label': self.config['labels'][str(int(logits.argmax(1).item()))],
                'g_text': gt.item(), 'g_image': gi.item(), 'disagreement': disagreement,
                'warning': policy['message'] if policy['enabled'] and disagreement >= policy['threshold'] else None,
                'gate_note': 'Learned coefficients are not definitive feature importance.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', default='gated_artifacts')
    parser.add_argument('--text', required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    with Image.open(args.image) as image:
        print(json.dumps(GatedPredictor(args.bundle).predict(args.text, image), indent=2))
