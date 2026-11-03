"""Fast OCR: RapidOCR (PP-OCRv6, ONNX) returning line-level boxes.

Line-level boxes on purpose. RapidOCR's `return_word_box` gives *word* boxes for
Latin text, and rebuilding spacing from them silently drops every space
(`def load(path)` → `defload(path)`); the recogniser's own line text keeps them.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return self.y1 - self.y0


_engine = None
_backend = "cpu"
_lock = threading.Lock()


def pick_backend() -> str:
    """`SCROLLBACK_GPU` = auto | dml | cuda | cpu. auto prefers CUDA, then DirectML."""
    want = os.environ.get("SCROLLBACK_GPU", "auto").lower()
    if want in ("cpu", "cuda", "dml"):
        return want
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return "cpu"
    if "CUDAExecutionProvider" in providers:
        return "cuda"
    if "DmlExecutionProvider" in providers:
        return "dml"
    return "cpu"


def _build(backend: str):
    from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    return RapidOCR(params={
        "Global.log_level": "error",
        "Global.use_cls": False,              # screen text is never rotated
        "Det.model_type": ModelType.TINY,
        "Det.lang_type": LangDet.CH,
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Rec.model_type": {"small": ModelType.SMALL, "medium": ModelType.MEDIUM,
                           "server": ModelType.SERVER}[os.environ.get("SCROLLBACK_REC", "medium")],
        "Rec.lang_type": LangRec.CH,          # the CH model reads ASCII code fine and Chinese comments too
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.rec_batch_num": 16,
        "EngineConfig.onnxruntime.use_cuda": backend == "cuda",
        "EngineConfig.onnxruntime.use_dml": backend == "dml",
        "EngineConfig.onnxruntime.intra_op_num_threads": 8,
    })


def _try_build(backend: str, timeout: float = 30.0):
    """GPU providers can hang while failing to initialise; never let that block the server."""
    box: dict = {}

    def run() -> None:
        try:
            eng = _build(backend)
            eng(np.zeros((64, 256, 3), dtype=np.uint8))
            box["eng"] = eng
        except Exception:  # noqa: BLE001
            pass

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    return box.get("eng")


def engine():
    global _engine, _backend
    with _lock:
        if _engine is None:
            backend = pick_backend()
            eng = _try_build(backend) if backend != "cpu" else None
            if eng is None:
                backend, eng = "cpu", _build("cpu")
            _engine, _backend = eng, backend
        return _engine


def backend() -> str:
    engine()
    return _backend


def recognize(crops: list[np.ndarray]) -> list[tuple[str, float]]:
    """Recognition only (no detection) on pre-cut single-line BGR crops, batched."""
    if not crops:
        return []
    res = engine().recognize_txt(crops)
    return [(t or "", float(s)) for t, s in zip(res.txts, res.scores)]


def read(bgr: np.ndarray, min_score: float = 0.3) -> list[Box]:
    """OCR a BGR image, one Box per detected text line."""
    res = engine()(bgr)
    if res.boxes is None:
        return []
    out = []
    for quad, text, score in zip(res.boxes, res.txts, res.scores):
        if not text or score < min_score:
            continue
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        out.append(Box(min(xs), min(ys), max(xs), max(ys), text, float(score)))
    return out
