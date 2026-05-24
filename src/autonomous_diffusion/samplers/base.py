"""Unified sampler interface.

Every baseline and the proposed method implement `Sampler.sample(...)` with the
same signature, so the run scripts can swap them by id without special-casing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass
class SamplerOutput:
    samples: torch.Tensor          # [N, C, H, W], in [-1, 1]
    nfe: int                       # network forward evals PER SAMPLE
    metadata: dict[str, Any] = field(default_factory=dict)


class Sampler(ABC):
    id: str = "abstract"

    @abstractmethod
    def sample(
        self,
        *,
        net: Any,
        num_samples: int,
        num_steps: int,
        seed: int,
        device: str | torch.device = "cuda",
        batch_size: int = 64,
        image_shape: tuple[int, int, int] | None = None,
    ) -> SamplerOutput: ...


_REGISTRY: dict[str, Callable[..., Sampler]] = {}
_ALIASES: dict[str, str] = {}   # alias_id -> canonical_id


def register_sampler(sampler_id: str):
    def deco(cls):
        if sampler_id in _REGISTRY:
            raise KeyError(f"sampler id {sampler_id!r} already registered")
        cls.id = sampler_id
        _REGISTRY[sampler_id] = cls
        return cls
    return deco


def register_alias(alias_id: str, canonical_id: str) -> None:
    """Make `alias_id` resolve to the same class as `canonical_id` via
    `get_sampler`, WITHOUT polluting `list_samplers()`. Used for backward-
    compat after a sampler rename."""
    if canonical_id not in _REGISTRY:
        raise KeyError(f"canonical sampler {canonical_id!r} not registered")
    _ALIASES[alias_id] = canonical_id


def get_sampler(sampler_id: str) -> Sampler:
    if sampler_id in _REGISTRY:
        inst = _REGISTRY[sampler_id]()
    elif sampler_id in _ALIASES:
        inst = _REGISTRY[_ALIASES[sampler_id]]()
        inst.id = sampler_id   # tag with requested id for trace clarity
    else:
        raise KeyError(f"unknown sampler {sampler_id!r}; have {list_samplers()}")
    return inst


def list_samplers() -> list[str]:
    """Canonical sampler ids only (aliases hidden)."""
    return sorted(_REGISTRY)
