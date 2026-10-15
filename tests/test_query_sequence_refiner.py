import importlib
import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn

from util.misc import NestedTensor


ROOT = Path(__file__).resolve().parents[1]


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QuerySequenceRefinerTest(unittest.TestCase):
    def setUp(self):
        self.dino_tool = importlib.import_module("models.dino.dino")

    def test_refiner_preserves_query_feature_shape(self):
        refiner = self.dino_tool.QuerySequenceRefiner(
            hidden_dim=16,
            num_layers=2,
            kernel_size=3,
            dropout=0.0,
        )
        query_feats = torch.randn(2, 5, 16)
        refined = refiner(query_feats)
        self.assertEqual(refined.shape, query_feats.shape)

    def test_model_without_refiner_does_not_create_refiner_module(self):
        model = self._build_fake_dino(use_query_sequence_refiner=False)
        self.assertFalse(model.use_query_sequence_refiner)
        self.assertFalse(hasattr(model, "query_sequence_refiner"))

    def test_model_with_refiner_emits_logits_and_boxes_with_same_query_count(self):
        model = self._build_fake_dino(
            use_query_sequence_refiner=True,
            query_sequence_refiner_layers=1,
            query_sequence_refiner_kernel=3,
        )
        samples = NestedTensor(
            torch.randn(2, 3, 8, 8),
            torch.zeros(2, 8, 8, dtype=torch.bool),
        )
        outputs = model(samples)
        self.assertTrue(model.use_query_sequence_refiner)
        self.assertEqual(outputs["pred_logits"].shape[:2], outputs["pred_boxes"].shape[:2])
        self.assertEqual(outputs["pred_logits"].shape, (2, 4, 8))

    def test_model_with_sorted_ctc_refiner_emits_sorted_ctc_logits(self):
        model = self._build_fake_dino(
            use_query_sequence_refiner=False,
            use_sorted_ctc_refiner=True,
            sorted_ctc_refiner_layers=1,
            sorted_ctc_refiner_kernel=3,
        )
        samples = NestedTensor(
            torch.randn(2, 3, 8, 8),
            torch.zeros(2, 8, 8, dtype=torch.bool),
        )
        outputs = model(samples)
        self.assertTrue(model.use_sorted_ctc_refiner)
        self.assertIn("pred_sorted_ctc_logits", outputs)
        self.assertEqual(outputs["pred_sorted_ctc_logits"].shape, outputs["pred_logits"].shape)
        self.assertEqual(outputs["pred_logits"].shape[:2], outputs["pred_boxes"].shape[:2])

    def test_vertical_geometry_adapter_is_exact_identity_at_zero_start(self):
        adapter = self.dino_tool.VerticalGeometryQueryAdapter(
            hidden_dim=16,
            geometry_hidden_dim=8,
            init_scale=0.0,
        )
        query_feats = torch.randn(2, 5, 16)
        sorted_boxes = torch.rand(2, 5, 4)
        sorted_boxes[:, :, 1], _ = torch.sort(sorted_boxes[:, :, 1], dim=1)
        adapted = adapter(query_feats, sorted_boxes, axis=1)
        self.assertTrue(torch.equal(adapted, query_feats))

    def test_model_with_vertical_geometry_adapter_emits_sorted_ctc_logits(self):
        model = self._build_fake_dino(
            use_query_sequence_refiner=False,
            use_vertical_geometry_adapter=True,
        )
        samples = NestedTensor(
            torch.randn(2, 3, 8, 8),
            torch.zeros(2, 8, 8, dtype=torch.bool),
        )
        outputs = model(samples)
        self.assertTrue(model.use_vertical_geometry_adapter)
        self.assertIn("pred_sorted_ctc_logits", outputs)
        self.assertEqual(outputs["pred_sorted_ctc_logits"].shape, outputs["pred_logits"].shape)

    def test_model_with_ctc_length_calibration_emits_initialized_length(self):
        model = self._build_fake_dino(
            use_query_sequence_refiner=False,
            use_ctc_length_calibration=True,
        )
        samples = NestedTensor(
            torch.randn(2, 3, 8, 8),
            torch.zeros(2, 8, 8, dtype=torch.bool),
        )
        outputs = model(samples)
        self.assertIn("pred_log_length", outputs)
        torch.testing.assert_close(
            torch.expm1(outputs["pred_log_length"]),
            torch.full((2,), 8.0),
        )

    def test_loss_ctc_uses_sorted_ctc_logits_when_available(self):
        criterion = self.dino_tool.SetCriterion(
            num_classes=2,
            matcher=None,
            weight_dict={"loss_CTC": 1.0},
            focal_alpha=0.25,
            losses=[],
            CTC=True,
            use_sorted_ctc_refiner=True,
        )
        outputs = {
            "pred_logits": torch.full((1, 2, 2), -8.0),
            "pred_boxes": torch.tensor([[[0.5, 0.9, 0.1, 0.1], [0.5, 0.1, 0.1, 0.1]]]),
            "pred_sorted_ctc_logits": torch.tensor([[[8.0, -8.0], [-8.0, 8.0]]]),
        }
        targets = [{"labels": torch.tensor([0]), "direction": torch.tensor(1)}]
        _, ctc_probs, _ = criterion.loss_CTC(
            outputs,
            targets,
            indices=None,
            num_boxes=None,
            return_preds=True,
        )
        self.assertGreater(ctc_probs[0, 0, 1].item(), 0.99)
        self.assertLess(ctc_probs[0, 0, 2].item(), 0.01)

    def _build_fake_dino(
        self,
        use_query_sequence_refiner,
        query_sequence_refiner_layers=1,
        query_sequence_refiner_kernel=3,
        use_sorted_ctc_refiner=False,
        sorted_ctc_refiner_layers=1,
        sorted_ctc_refiner_kernel=3,
        use_vertical_geometry_adapter=False,
        use_ctc_length_calibration=False,
    ):
        class FakeDecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.label_embedding = None

        class FakeDecoder(nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.layers = nn.ModuleList([FakeDecoderLayer() for _ in range(num_layers)])
                self.bbox_embed = None
                self.class_embed = None

        class FakeTransformer(nn.Module):
            def __init__(self, d_model=32, num_decoder_layers=1, num_queries=4):
                super().__init__()
                self.d_model = d_model
                self.num_decoder_layers = num_decoder_layers
                self.num_queries = num_queries
                self.decoder = FakeDecoder(num_decoder_layers)

            def forward(self, srcs, masks, input_query_bbox, poss, input_query_label, attn_mask):
                batch_size = srcs[0].shape[0]
                hs = torch.randn(self.num_decoder_layers, batch_size, self.num_queries, self.d_model)
                reference = torch.full((self.num_decoder_layers + 1, batch_size, self.num_queries, 4), 0.5)
                return hs, reference, None, None, None

        class FakeBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_channels = [3]

            def __getitem__(self, index):
                return self

            def forward(self, samples):
                feature = NestedTensor(samples.tensors, samples.mask)
                pos = torch.zeros(
                    samples.tensors.shape[0],
                    32,
                    samples.tensors.shape[2],
                    samples.tensors.shape[3],
                )
                return [feature], [pos]

        return self.dino_tool.DINO(
            backbone=FakeBackbone(),
            transformer=FakeTransformer(),
            num_classes=8,
            num_queries=4,
            aux_loss=False,
            iter_update=True,
            query_dim=4,
            num_feature_levels=1,
            two_stage_type="no",
            use_query_sequence_refiner=use_query_sequence_refiner,
            query_sequence_refiner_layers=query_sequence_refiner_layers,
            query_sequence_refiner_kernel=query_sequence_refiner_kernel,
            use_sorted_ctc_refiner=use_sorted_ctc_refiner,
            sorted_ctc_refiner_layers=sorted_ctc_refiner_layers,
            sorted_ctc_refiner_kernel=sorted_ctc_refiner_kernel,
            use_vertical_geometry_adapter=use_vertical_geometry_adapter,
            use_ctc_length_calibration=use_ctc_length_calibration,
            ctc_length_init=8.0,
        )


class QuerySequenceRefinerConfigSmokeTest(unittest.TestCase):
    def test_mthv2_sqr_probe_config_sets_expected_flags(self):
        config = load_module(ROOT / "config" / "MTHV2_dtlr_sqr_probe.py")
        self.assertTrue(config.use_query_sequence_refiner)
        self.assertEqual(config.query_sequence_refiner_layers, 1)
        self.assertEqual(config.query_sequence_refiner_kernel, 3)
        self.assertEqual(config.query_sequence_refiner_dropout, 0.0)
        self.assertEqual(config.forced_direction, "vertical")

    def test_mthv2_sorted_ctc_refiner_probe_config_sets_expected_flags(self):
        config = load_module(ROOT / "config" / "MTHV2_dtlr_sorted_ctc_refiner_probe.py")
        self.assertTrue(config.use_sorted_ctc_refiner)
        self.assertEqual(config.sorted_ctc_refiner_axis, 1)
        self.assertEqual(config.sorted_ctc_refiner_layers, 1)
        self.assertEqual(config.sorted_ctc_refiner_kernel, 3)
        self.assertEqual(config.forced_direction, "vertical")

    def test_mthv2_vertical_geometry_adapter_probe_config(self):
        config = load_module(
            ROOT / "config" / "MTHV2_dtlr_vertical_geometry_adapter_probe.py"
        )
        self.assertTrue(config.use_vertical_geometry_adapter)
        self.assertEqual(config.vertical_geometry_adapter_axis, 1)
        self.assertEqual(config.vertical_geometry_adapter_hidden_dim, 64)
        self.assertEqual(config.vertical_geometry_adapter_init_scale, 0.0)
        self.assertEqual(config.forced_direction, "vertical")


if __name__ == "__main__":
    unittest.main()
