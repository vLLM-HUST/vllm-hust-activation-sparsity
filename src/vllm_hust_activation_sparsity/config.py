# SPDX-License-Identifier: Apache-2.0
"""Fail-closed activation-sparsity configuration from legacy PR #67."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

SUPPORTED_TARGET_PROJECTIONS = frozenset(
    {"self_attn.qkv", "self_attn.o", "mlp.gate_up", "mlp.down"}
)


@dataclass(frozen=True)
class ActivationSparsityConfig:
    enable: bool = False
    method: str = "teal"
    uniform_sparsity: float = 0.0
    calibration_path: str | None = None
    decode_only: bool = False
    apply_all_tokens: bool = False
    prefill_sparsify: str = "half"
    use_sparse_gemv: bool = False
    target_projections: tuple[str, ...] | None = None
    target_layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.method not in {"teal", "larosa"}:
            raise ValueError("method must be 'teal' or 'larosa'")
        if self.prefill_sparsify not in {"half", "all", "none"}:
            raise ValueError("prefill_sparsify must be half, all, or none")
        if not math.isfinite(self.uniform_sparsity):
            raise ValueError("uniform_sparsity must be finite")
        if not 0.0 <= self.uniform_sparsity < 1.0:
            raise ValueError("uniform_sparsity must be in [0, 1)")
        if self.method == "larosa" and self.uniform_sparsity * 1.2 >= 1.0:
            raise ValueError("La RoSA uniform_sparsity must be lower than 5/6")
        if self.use_sparse_gemv and not self.enable:
            raise ValueError("use_sparse_gemv requires enable=True")
        if self.target_projections is not None:
            unknown = sorted(
                set(self.target_projections) - SUPPORTED_TARGET_PROJECTIONS
            )
            if unknown:
                raise ValueError(f"unsupported target projections: {unknown}")
        if self.target_layers is not None and any(
            type(layer) is not int or layer < 0 for layer in self.target_layers
        ):
            raise ValueError("target_layers must be non-negative integers")

    def compute_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


def validate_compatibility(
    config: ActivationSparsityConfig,
    *,
    tensor_parallel_size: int,
    has_quantization: bool,
    has_lora: bool,
) -> None:
    if not config.enable:
        return
    if not config.calibration_path:
        raise ValueError("enabled activation sparsity requires calibration_path")
    if tensor_parallel_size != 1:
        raise ValueError("activation sparsity requires tensor_parallel_size=1")
    if has_quantization:
        raise ValueError("activation sparsity does not support quantization")
    if has_lora:
        raise ValueError("activation sparsity does not support LoRA")


__all__ = [
    "ActivationSparsityConfig",
    "SUPPORTED_TARGET_PROJECTIONS",
    "validate_compatibility",
]
