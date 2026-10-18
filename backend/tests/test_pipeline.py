"""Çalıştırma (backend/ klasöründen):  python -m unittest discover -s tests"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app import signals
from app.inference import ExportedModel, PhysiologicalModel, reconstruct_physiological
from app.leads import WINDOW_SAMPLES
from app.preprocessing import bandpass, lead_mask, preprocess


class PreprocessingTest(unittest.TestCase):
    def test_removes_baseline_wander_keeps_qrs_band(self):
        t = np.arange(WINDOW_SAMPLES) / 100
        wander = np.sin(2 * np.pi * 0.1 * t)
        band = np.sin(2 * np.pi * 10 * t)
        out = bandpass(np.tile(wander + band, (12, 1)))[0, 200:-200]
        self.assertLess(np.abs(out - band[200:-200]).max(), 0.05)

    def test_masked_leads_stay_zero(self):
        raw = signals.generate(0, WINDOW_SAMPLES, "normal")
        mask = lead_mask([2, 7])
        filtered, model_input = preprocess(raw * mask[:, None], mask)
        self.assertEqual(model_input.shape, (12, WINDOW_SAMPLES))
        self.assertTrue(np.all(model_input[[2, 7]] == 0))
        self.assertAlmostEqual(float(model_input[0].std()), 1.0, places=3)


class ReconstructionTest(unittest.TestCase):
    def test_einthoven_is_near_exact_for_limb_leads(self):
        raw = signals.generate(0, WINDOW_SAMPLES, "inf")
        mask = lead_mask([2, 3])
        rec, methods = reconstruct_physiological(raw * mask[:, None], mask)
        self.assertEqual(methods, {2: "einthoven", 3: "einthoven"})
        for lead in (2, 3):
            self.assertGreater(np.corrcoef(rec[lead], raw[lead])[0, 1], 0.95)

    def test_precordial_interpolation(self):
        raw = signals.generate(0, WINDOW_SAMPLES, "normal")
        mask = lead_mask([8])
        rec, methods = reconstruct_physiological(raw * mask[:, None], mask)
        self.assertEqual(methods[8], "interpolation")
        self.assertGreater(np.corrcoef(rec[8], raw[8])[0, 1], 0.9)


class PhysiologicalModelTest(unittest.TestCase):
    model = PhysiologicalModel()

    def run_case(self, condition, disconnected=()):
        raw = signals.generate(0, WINDOW_SAMPLES, condition)
        mask = lead_mask(list(disconnected))
        return self.model.predict(raw * mask[:, None], mask).to_json()

    def test_detects_each_wall(self):
        for condition, region in [("normal", None), ("ant", "anterior"), ("inf", "inferior"), ("lat", "lateral")]:
            with self.subTest(condition=condition):
                result = self.run_case(condition)
                self.assertEqual(result["region"], region)
                if region:
                    self.assertGreater(result["stDeviation"], 1.5)

    def test_robust_to_up_to_four_missing_leads_outside_the_wall(self):
        self.assertEqual(self.run_case("inf", [6, 7, 8, 9])["region"], "inferior")
        self.assertEqual(self.run_case("lat", [1, 2, 5, 6])["region"], "lateral")

    def test_confidence_drops_when_leads_missing(self):
        full = self.run_case("inf")["confidence"]
        partial = self.run_case("inf", [1, 2])["confidence"]
        self.assertLess(partial, full)

    def test_attention_ignores_missing_leads(self):
        result = self.run_case("ant", [6])
        self.assertEqual(result["attention"][6], 0)
        self.assertAlmostEqual(sum(result["attention"]), 1.0, places=3)


class ChunkEndpointTest(unittest.TestCase):
    def test_returns_requested_length_and_reconstruction(self):
        from app.main import ChunkRequest, chunk

        for start, count in [(0, 1100), (5000, 1100), (5000, 5)]:
            response = chunk(ChunkRequest(start=start, count=count, disconnectedLeads=[2]))
            self.assertEqual(len(response["data"][0]), count)
            self.assertEqual(len(response["reconstructed"]["2"]), count)
            self.assertTrue(all(v == 0 for v in response["data"][2]))


class ExportedModelContractTest(unittest.TestCase):
    """İP3 modelinin TorchScript dışa aktarım sözleşmesini sahte bir modelle doğrular."""

    def test_torchscript_contract(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch kurulu değil")

        class Dummy(torch.nn.Module):
            def forward(self, x, mask):
                logits = torch.tensor([[0.0, 0.0, 3.0, 0.0]])
                return x, logits, mask / mask.sum()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mae.pt"
            torch.jit.script(Dummy()).save(str(path))
            path.with_suffix(".stats.json").write_text(json.dumps({"std": [0.3] * 12}))
            model = ExportedModel(path)
            raw = signals.generate(0, WINDOW_SAMPLES, "inf")
            mask = lead_mask([4])
            result = model.predict(raw * mask[:, None], mask).to_json()
            self.assertEqual(result["region"], "inferior")
            self.assertEqual(result["reconstruction"], {"4": "mae"})
            self.assertTrue(result["modelSource"].startswith("pt:"))


if __name__ == "__main__":
    unittest.main()
