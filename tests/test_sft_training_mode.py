from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from opensearch_vl_repro.sft_long_training import (
    activate_sft_training_mode,
    run_sft_stage,
)


class FakeSFTModel:
    def __init__(self):
        self.training = False
        self.is_gradient_checkpointing = True
        self.language_model = SimpleNamespace(
            training=False,
            gradient_checkpointing=True,
            layers=[SimpleNamespace(training=False, gradient_checkpointing=True)
                    for _ in range(36)],
        )

    def train(self):
        self.training = True
        self.language_model.training = True
        for layer in self.language_model.layers:
            layer.training = True
        return self

    def named_modules(self):
        yield "", self
        yield "base_model.model.model.language_model", self.language_model


def test_sft_training_mode_activates_all_decoder_layers():
    model = FakeSFTModel()
    activate_sft_training_mode(model)
    assert model.training and model.language_model.training
    assert len(model.language_model.layers) == 36
    assert all(layer.training and layer.gradient_checkpointing
               for layer in model.language_model.layers)


def test_sft_training_mode_rejects_silent_checkpointing_failures():
    model = FakeSFTModel()
    def train_wrapper_only():
        model.training = True  # Reproduce the observed PEFT/inner-model mismatch.
        return model
    model.train = train_wrapper_only
    with pytest.raises(RuntimeError, match="training mode"):
        activate_sft_training_mode(model)

    model = FakeSFTModel()
    def train_except_one_layer():
        FakeSFTModel.train(model)
        model.language_model.layers[21].training = False
        return model
    model.train = train_except_one_layer
    with pytest.raises(RuntimeError, match="training mode"):
        activate_sft_training_mode(model)

    model = FakeSFTModel()
    model.language_model.layers[21].gradient_checkpointing = False
    with pytest.raises(RuntimeError, match="gradient checkpointing enabled"):
        activate_sft_training_mode(model)


def test_formal_sft_activates_training_mode_before_ddp():
    source = inspect.getsource(run_sft_stage)
    assert source.index("activate_sft_training_mode(model)") < source.index(
        "DistributedDataParallel(model")
