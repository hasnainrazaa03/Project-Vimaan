"""Tests for the ONNX inference backend (Phase 4C).

These exercise the duck-typing contract without needing a real exported graph:
``OnnxBackend`` must accept torch tensors, feed int64 numpy to the session, and
return ``(None, intent_logits, slot_logits)`` as torch tensors so that
``predict()`` and the plugin handlers keep working unchanged.
"""

import numpy as np
import pytest

requires_torch = pytest.mark.requires_torch


class _FakeSession:
    """Minimal stand-in for onnxruntime.InferenceSession."""

    def __init__(self, intent, slot):
        self._intent = intent
        self._slot = slot
        self.last_feeds = None

    def run(self, names, feeds):
        self.last_feeds = feeds
        assert names == ["intent_logits", "slot_logits"]
        return [self._intent, self._slot]


@requires_torch
def test_onnx_model_path_layout():
    from vimaan_nlu.onnx_backend import ONNX_FILENAME, ONNX_SUBDIR, onnx_model_path

    path = onnx_model_path("/tmp/models/v9")
    assert path.endswith(f"{ONNX_SUBDIR}/{ONNX_FILENAME}")
    assert path == "/tmp/models/v9/onnx/model.onnx"


@requires_torch
def test_backend_returns_torch_tensors_and_none_loss():
    import torch
    from vimaan_nlu.onnx_backend import OnnxBackend

    intent = np.array([[0.1, 0.9, -0.3]], dtype=np.float32)
    slot = np.zeros((1, 4, 5), dtype=np.float32)
    backend = OnnxBackend(session=_FakeSession(intent, slot))

    ids = torch.ones(1, 4, dtype=torch.long)
    mask = torch.ones(1, 4, dtype=torch.long)
    loss, intent_logits, slot_logits = backend(ids, mask)

    assert loss is None
    assert isinstance(intent_logits, torch.Tensor)
    assert isinstance(slot_logits, torch.Tensor)
    assert intent_logits.shape == (1, 3)
    assert slot_logits.shape == (1, 4, 5)
    assert intent_logits.argmax(1).item() == 1


@requires_torch
def test_backend_feeds_int64_numpy():
    import torch
    from vimaan_nlu.onnx_backend import OnnxBackend

    session = _FakeSession(
        np.zeros((1, 2), dtype=np.float32), np.zeros((1, 3, 2), dtype=np.float32)
    )
    backend = OnnxBackend(session=session)

    # Feed int32 tensors; backend must up-cast to int64 for onnxruntime.
    ids = torch.ones(1, 3, dtype=torch.int32)
    backend(ids, ids)

    assert session.last_feeds["input_ids"].dtype == np.int64
    assert session.last_feeds["attention_mask"].dtype == np.int64
    assert set(session.last_feeds) == {"input_ids", "attention_mask"}


@requires_torch
def test_backend_eval_and_to_are_noops():
    from vimaan_nlu.onnx_backend import OnnxBackend

    backend = OnnxBackend(
        session=_FakeSession(
            np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1, 1), dtype=np.float32)
        )
    )
    assert backend.eval() is backend
    assert backend.to("cpu") is backend


@requires_torch
def test_missing_onnx_file_raises():
    from vimaan_nlu.onnx_backend import OnnxBackend

    with pytest.raises(FileNotFoundError):
        OnnxBackend(onnx_path="/no/such/model.onnx")
