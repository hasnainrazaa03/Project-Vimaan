"""ONNX Runtime inference backend for the joint intent+slot model (Phase 4C).

The backend *duck-types* the PyTorch model: it is callable as
``backend(input_ids, attention_mask)`` and returns
``(None, intent_logits, slot_logits)`` as ``torch.Tensor`` objects, exactly
like ``JointIntentAndSlotModel.forward`` in eval mode. That means
``vimaan_nlu.inference.predict`` and every plugin handler work unchanged no
matter which backend is loaded -- only ``ModelLoader`` decides which one to
build.

``onnxruntime`` is an optional dependency; importing this module does not
require it. The session is created lazily in ``__init__`` so unit tests can
inject a fake session instead.
"""

import os

import torch

ONNX_SUBDIR = "onnx"
ONNX_FILENAME = "model.onnx"


def onnx_model_path(model_path):
    """Return the conventional ``<model_path>/onnx/model.onnx`` location."""
    return os.path.join(model_path, ONNX_SUBDIR, ONNX_FILENAME)


class OnnxBackend:
    """Callable wrapper around an ``onnxruntime.InferenceSession``.

    Parameters
    ----------
    onnx_path:
        Path to the exported ``model.onnx``. Ignored when ``session`` is given.
    session:
        Pre-built session (used by tests). When ``None`` a real
        ``onnxruntime.InferenceSession`` is created from ``onnx_path``.
    """

    def __init__(self, onnx_path=None, session=None, providers=None):
        if session is not None:
            self.session = session
        else:
            if not onnx_path or not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                onnx_path, providers=providers or ["CPUExecutionProvider"]
            )
        # quantized/onnx inference is CPU-only in this project
        self.device = torch.device("cpu")

    def __call__(self, input_ids, attention_mask, intent_labels=None, slot_labels=None):
        feeds = {
            "input_ids": _to_int64_numpy(input_ids),
            "attention_mask": _to_int64_numpy(attention_mask),
        }
        intent_logits, slot_logits = self.session.run(["intent_logits", "slot_logits"], feeds)
        return None, torch.from_numpy(intent_logits), torch.from_numpy(slot_logits)

    # ``predict`` calls model.eval()/.to() in some paths; make them no-ops.
    def eval(self):
        return self

    def to(self, _device):
        return self


def _to_int64_numpy(tensor):
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    arr = tensor.numpy() if hasattr(tensor, "numpy") else tensor
    if arr.dtype != "int64":
        arr = arr.astype("int64")
    return arr
