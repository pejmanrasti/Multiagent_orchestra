"""
Checks for the framework's image input adapter (preprocess.py). Needs torch +
torchvision, so it is skipped in an environment without them, e.g. run it with
an image env's python:

    .envs/<env>/bin/python -m unittest tests.test_preprocess -v
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None and \
    importlib.util.find_spec("torchvision") is not None


def _load_preprocess(tmp: Path):
    from workspace_files import PREPROCESS_PY
    (tmp / "preprocess.py").write_text(PREPROCESS_PY)
    sys.path.insert(0, str(tmp))
    sys.modules.pop("preprocess", None)
    import preprocess
    return preprocess


@unittest.skipUnless(HAS_TORCH, "torch/torchvision not installed in this interpreter")
class TestAdaptiveInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pp = _load_preprocess(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls.tmp.name)
        cls.tmp.cleanup()

    def _probe(self):
        """A network that records what it receives."""
        torch, nn = self.torch, self.torch.nn
        seen = {}

        class Probe(nn.Module):
            def forward(self, x):
                seen["x"] = x
                return torch.zeros(x.shape[0], 3)
        return Probe(), seen

    def test_resizes_to_model_size(self):
        net, seen = self._probe()
        m = self.pp.prepare_model(net, image_size=64)
        m(self.torch.rand(2, 3, 32, 48))
        self.assertEqual(tuple(seen["x"].shape), (2, 3, 64, 64))

    def test_channels_range_and_normalisation(self):
        torch = self.torch
        net, seen = self._probe()
        m = self.pp.prepare_model(net, image_size=8, mean=[0.5] * 3, std=[0.25] * 3)
        m(torch.full((1, 1, 8, 8), 255.0))                      # grayscale, 0-255
        self.assertEqual(seen["x"].shape[1], 3)
        self.assertTrue(torch.allclose(seen["x"], torch.full_like(seen["x"], 2.0)))  # (1-.5)/.25
        m(torch.rand(3, 8, 8))                                  # unbatched
        self.assertEqual(seen["x"].shape[0], 1)
        with self.assertRaisesRegex(ValueError, "already normalised"):
            m(torch.randn(1, 3, 8, 8) - 2)
        with self.assertRaisesRegex(ValueError, "channels"):
            m(torch.rand(1, 2, 8, 8))

    def test_spec_from_pretrained_weights_fixes_vit_size(self):
        """ViT-B/16 only accepts 224px; data loaded at 160px used to crash."""
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        torch = self.torch
        spec = self.pp.make_spec(weights=ViT_B_16_Weights.DEFAULT)
        self.assertEqual(spec["image_size"], 224)
        net = vit_b_16(weights=None, num_classes=3)      # same architecture, no download
        with self.assertRaises(Exception):
            net(torch.rand(1, 3, 160, 160))
        model = self.pp.prepare_model(net, weights=ViT_B_16_Weights.DEFAULT)
        loader = [(torch.rand(2, 3, 160, 160), torch.tensor([0, 1]))]
        self.pp.check_model(model, loader, 3)             # passes thanks to the resize
        self.assertEqual(tuple(model(torch.rand(1, 3, 160, 160)).shape), (1, 3))

    def test_check_model_reports_problems(self):
        torch, nn = self.torch, self.torch.nn
        loader = [(torch.rand(2, 3, 16, 16), torch.tensor([0, 1]))]
        net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 5))
        with self.assertRaisesRegex(ValueError, "expected \\(batch, 3\\)"):
            self.pp.check_model(self.pp.prepare_model(net, image_size=16), loader, 3)
        with self.assertRaisesRegex(TypeError, "prepare_model"):
            self.pp.check_model(net, loader, 5)
        bad = nn.Sequential(nn.Flatten(), nn.Linear(10, 3))     # wrong input features
        with self.assertRaisesRegex(RuntimeError, "sample batch"):
            self.pp.check_model(self.pp.prepare_model(bad, image_size=16), loader, 3)

    def test_saved_model_keeps_preprocessing(self):
        torch, nn = self.torch, self.torch.nn
        net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
        m = self.pp.prepare_model(net, image_size=20, mean=[0.1, 0.2, 0.3], std=[1, 1, 1])
        f = Path(self.tmp.name) / "m.pt"
        torch.save(m, f)
        loaded = torch.load(f, weights_only=False)
        self.assertIsInstance(loaded, self.pp.AdaptiveInput)
        self.assertEqual(loaded.input_spec, m.input_spec)
        x = torch.rand(1, 3, 40, 40)
        self.assertTrue(torch.allclose(loaded(x), m(x)))


if __name__ == "__main__":
    unittest.main()
