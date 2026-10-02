"""Purity checks for exported students (CLAUDE.md Rule #1: no quantum ops in the exported
model, ever). Every check raises PurityError with the offending items listed — this is a
gate, so it fails loudly rather than warning.

Applied at three stages, because each can introduce something the previous could not see:
  1. the PyTorch checkpoint  — only student weights, and only standard nn modules
  2. the ONNX graph          — only standard-domain ops from a known set
  3. the INT8 TFLite model   — only builtin ops, every tensor integer (no float fallback),
                               and per-channel weights on every conv (Trap #5)
"""
import numpy as np


class PurityError(AssertionError):
    pass


# Anything training-only that must never reach an exported artefact. The kernel projection
# heads and the teacher are separate modules in the trainer and are not saved with the
# student — this guards against that ever changing.
_FORBIDDEN_KEY_PARTS = ("proj", "kernel", "quantum", "qnode", "teacher", "pennylane")

_ALLOWED_TORCH_MODULES = {
    "AdaptiveAvgPool2d", "BatchNorm2d", "Conv2d", "Conv2dNormActivation", "Dropout",
    "Hardsigmoid", "Hardswish", "Identity", "InvertedResidual", "Linear", "ReLU", "ReLU6",
    "Sequential", "SqueezeExcitation", "MaxPool2d", "Flatten",
    # the student wrapper classes themselves
    "LeNet5", "MobileNetV2Student", "MobileNetV3SmallStudent",
}

_ALLOWED_ONNX_OPS = {
    "Add", "AveragePool", "BatchNormalization", "Clip", "Concat", "Constant", "Conv", "Div",
    "Flatten", "Gather", "Gemm", "GlobalAveragePool", "HardSigmoid", "HardSwish", "Identity",
    "MatMul", "MaxPool", "Mul", "ReduceMean", "Relu", "Reshape", "Shape", "Sigmoid",
    "Squeeze", "Sub", "Unsqueeze",
}

# Tensor dtypes allowed in a *fully integer* TFLite model: int8 activations/weights and
# int32 for biases and shape operands. Any float32 tensor means a float op survived.
_INTEGER_DTYPES = {np.int8, np.int32}


def check_state_dict(state_dict):
    bad = [k for k in state_dict if any(p in k.lower() for p in _FORBIDDEN_KEY_PARTS)]
    if bad:
        raise PurityError(f"checkpoint contains training-only parameters: {bad[:10]}")


def check_torch_modules(model):
    found = {type(m).__name__ for m in model.modules()}
    bad = sorted(found - _ALLOWED_TORCH_MODULES)
    if bad:
        raise PurityError(f"student contains non-allowlisted modules: {bad}")


def check_onnx(onnx_model):
    bad_domain = sorted({n.domain for n in onnx_model.graph.node if n.domain not in ("", "ai.onnx")})
    if bad_domain:
        raise PurityError(f"ONNX graph uses custom op domains: {bad_domain}")
    ops = {n.op_type for n in onnx_model.graph.node}
    bad = sorted(ops - _ALLOWED_ONNX_OPS)
    if bad:
        raise PurityError(f"ONNX graph contains non-allowlisted ops: {bad}")
    return sorted(ops)


def check_tflite(interpreter):
    """Returns a summary dict; raises PurityError on any violation."""
    ops = interpreter._get_ops_details()
    op_names = [o["op_name"] for o in ops]

    custom = sorted({n for n in op_names if n.upper().startswith(("CUSTOM", "FLEX"))})
    if custom:
        raise PurityError(f"TFLite model contains custom/flex ops: {custom}")
    if "DEQUANTIZE" in op_names:
        raise PurityError("TFLite model contains DEQUANTIZE — a float section survived conversion")

    tensors = {t["index"]: t for t in interpreter.get_tensor_details()}
    float_tensors = [t["name"] for t in tensors.values() if np.dtype(t["dtype"]).type not in _INTEGER_DTYPES]
    if float_tensors:
        raise PurityError(f"model is not fully integer — {len(float_tensors)} non-integer tensors, "
                          f"e.g. {float_tensors[:5]}")

    # Trap #5: depthwise convs need per-channel weight quantization. A conv weight with a
    # single scale is per-tensor and will cost accuracy that looks like a method failure.
    per_tensor = []
    n_conv = 0
    for o in ops:
        if o["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            n_conv += 1
            w = tensors[o["inputs"][1]]
            if len(w["quantization_parameters"]["scales"]) <= 1:
                per_tensor.append((o["op_name"], w["name"]))
    if per_tensor:
        raise PurityError(f"{len(per_tensor)} conv weights are per-tensor, not per-channel: {per_tensor[:5]}")

    counts = {}
    for n in op_names:
        counts[n] = counts.get(n, 0) + 1
    return {"ops": counts, "n_conv": n_conv, "fully_integer": True, "per_channel": True}
