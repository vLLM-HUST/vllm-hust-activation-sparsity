import pytest

from vllm_hust_activation_sparsity import (
    ActivationSparsityConfig,
    validate_compatibility,
)


def test_config_hash_is_stable_and_sensitive() -> None:
    left = ActivationSparsityConfig(enable=True, calibration_path="calibration")
    right = ActivationSparsityConfig(enable=True, calibration_path="calibration")
    changed = ActivationSparsityConfig(
        enable=True, calibration_path="calibration", uniform_sparsity=0.4
    )
    assert left.compute_hash() == right.compute_hash()
    assert left.compute_hash() != changed.compute_hash()


def test_unsupported_combinations_fail_closed() -> None:
    config = ActivationSparsityConfig(enable=True, calibration_path="calibration")
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        validate_compatibility(
            config, tensor_parallel_size=2, has_quantization=False, has_lora=False
        )
    with pytest.raises(ValueError, match="quantization"):
        validate_compatibility(
            config, tensor_parallel_size=1, has_quantization=True, has_lora=False
        )


def test_larosa_and_projection_validation() -> None:
    with pytest.raises(ValueError, match="5/6"):
        ActivationSparsityConfig(method="larosa", uniform_sparsity=0.9)
    with pytest.raises(ValueError, match="unsupported target"):
        ActivationSparsityConfig(target_projections=("unknown",))
