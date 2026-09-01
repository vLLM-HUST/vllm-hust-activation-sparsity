"""Activation sparsity configuration and inert runtime metadata."""

from .config import (
    SUPPORTED_TARGET_PROJECTIONS,
    ActivationSparsityConfig,
    validate_compatibility,
)


class VllmHustActivationSparsityContractProposal:
    """Metadata-only proposal; this class performs no runtime activation."""


__all__ = [
    "ActivationSparsityConfig",
    "SUPPORTED_TARGET_PROJECTIONS",
    "VllmHustActivationSparsityContractProposal",
    "validate_compatibility",
]
