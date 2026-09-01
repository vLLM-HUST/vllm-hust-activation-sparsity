# Activation sparsity host contract proposal

The extracted configuration preserves TEAL/La RoSA validation and compilation
cache identity without importing vLLM or torch. Runtime activation requires:

1. `vllm.model.projection-transform.v1`: register a projection transform by
   canonical layer/projection name without monkey-patching model classes.
2. `vllm.model.weight-loader-transform.v1`: optionally merge a validated La RoSA
   rotation during weight loading with shape and artifact checks.
3. `vllm.compilation.config-hash.v1`: include the immutable sparsity config hash
   in compilation/cache identity.
4. `vllm.operator.sparse-linear.v1`: optional device kernel capability query and
   dense fallback; unsupported TP, quantization, and LoRA combinations fail closed.

The extension must not modify model weights after loading, silently select an
experimental kernel, or download calibration data at import time. Ascend/CUDA
kernels belong in separate device-provider components with their own version and
hardware evidence.
