"""Tiny CPU PEFT-shaped 36x7 fixture, never real model/GPU evidence."""
from types import SimpleNamespace

import torch
from opensearch_vl_repro.rl.rl_actor_semantics import LORA_TARGETS


def sft_config():
    return {"lora": dict(rank=16, alpha=32, dropout=.05, target_modules=list(LORA_TARGETS))}


def target():
    module = torch.nn.Module()
    module.active_adapters = ["default"]
    module._disable_adapters = False
    module.merged_adapters = []
    module.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1, bias=False)})
    module.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1, bias=False)})
    module.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(.05)})
    with torch.no_grad():
        module.lora_A["default"].weight.fill_(1.)
        module.lora_B["default"].weight.fill_(1.)
    return module


def model():
    value = torch.nn.Module()
    value.active_adapters = ["default"]
    value._adapters_disabled = False
    value.peft_config = {"default": SimpleNamespace(r=16, lora_alpha=32, lora_dropout=.05, target_modules=list(LORA_TARGETS))}
    value.visual = torch.nn.Module()
    value.visual.merger = torch.nn.Linear(1, 1)
    value.base = torch.nn.Linear(1, 1)
    for p in value.parameters():
        p.requires_grad_(False)
    value.language_model = torch.nn.Module()
    value.language_model.layers = torch.nn.ModuleList()
    for _ in range(36):
        layer = torch.nn.Module()
        layer.gradient_checkpointing = True
        layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
        for name in LORA_TARGETS:
            setattr(layer.self_attn if name in LORA_TARGETS[:4] else layer.mlp, name, target())
        value.language_model.layers.append(layer)
    value.forward = lambda x: x * first_weight(value).reshape(())
    return value


def first_target(value):
    return value.language_model.layers[0].self_attn.q_proj


def first_weight(value):
    return first_target(value).lora_B["default"].weight


def actor():
    value = model()
    return SimpleNamespace(actor_module=value, actor_optimizer=torch.optim.AdamW(
        [p for p in value.parameters() if p.requires_grad], lr=1e-6))
