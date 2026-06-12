import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from ..airllm.airllm_base import AirLLMBaseModel


class DummyEmbed(nn.Module):
    def forward(self, x):
        return x.float()


class DummyBlock(nn.Module):
    def forward(self, x, **kwargs):
        return (x + 1.0,)


class DummyNorm(nn.Module):
    def forward(self, x):
        return x


class DummyHead(nn.Module):
    def forward(self, x):
        return x


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = DummyEmbed()
        self.layers = nn.ModuleList([DummyBlock(), DummyBlock()])
        self.norm = DummyNorm()


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyBackbone()
        self.lm_head = DummyHead()

    def tie_weights(self):
        return None


class DummyAirLLM(AirLLMBaseModel):
    def init_model(self):
        self.model = DummyModel()
        self.model.eval()
        self.set_layers_from_layer_names()

    def get_generation_config(self):
        return SimpleNamespace()

    def get_tokenizer(self, hf_token=None):
        class _Tok:
            pass

        return _Tok()

    def load_layer_to_cpu(self, layer_name):
        # Keep forward lightweight by avoiding disk/model state loading.
        return {}

    def move_layer_to_device(self, state_dict):
        # No tensor movement in dummy tests; just satisfy the interface.
        return []


class TestMemorySafetyClamp(unittest.TestCase):
    def _build_model(self, max_layers_in_memory):
        with mock.patch(
            "air_llm.airllm.airllm_base.find_or_create_local_splitted_path",
            return_value=(Path("/tmp"), Path("/tmp")),
        ), mock.patch(
            "air_llm.airllm.airllm_base.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(),
        ):
            return DummyAirLLM(
                model_local_path_or_repo_id="dummy/repo",
                device="cpu",
                prefetching=False,
                max_layers_in_memory=max_layers_in_memory,
            )

    def test_init_clamps_requested_layers_and_warns(self):
        with mock.patch.object(
            DummyAirLLM,
            "detect_max_layers_in_memory",
            return_value=3,
        ):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                model = self._build_model(max_layers_in_memory=999)

        output = buf.getvalue()
        self.assertEqual(model.max_layers_in_memory, 3)
        self.assertIn("WARNING: requested max_layers_in_memory=999 exceeds safe limit=3", output)
        self.assertIn("Clamping to 3.", output)

    def test_forward_runtime_clamps_and_warns(self):
        with mock.patch.object(
            DummyAirLLM,
            "detect_max_layers_in_memory",
            return_value=2,
        ):
            model = self._build_model(max_layers_in_memory=1)

        # Simulate user overriding the value after init.
        model.max_layers_in_memory = 999

        # Force runtime safe limit lower than requested to trigger warning.
        with mock.patch.object(model, "detect_max_layers_in_memory", return_value=1):
            input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _ = model(input_ids=input_ids, use_cache=False, return_dict=False)

        output = buf.getvalue()
        self.assertIn("runtime memory pressure reduced safe layer chunk size to 1", output)
        self.assertIn("instead of requested 999", output)


if __name__ == "__main__":
    unittest.main()