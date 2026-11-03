"""Pick the frames worth reading from a screen recording.

A frame is read when the screen has just come to rest (after a page-down, a tab
switch …), and — while it keeps moving, e.g. continuous wheel scrolling — at a
fixed cadence so nothing scrolls past unseen.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Keyframe:
    idx: int
    t: float
    bgr: np.ndarray
    moving: bool


def info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = {"fps": fps, "frames": n, "seconds": round(n / fps, 1),
           "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return out


def frame_at(path: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise IndexError(f"frame {idx} not readable in {path}")
    return bgr


def _sig(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    return cv2.resize(g, (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.int16)


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    """Number of clearly changed pixels on a quarter-size thumbnail.

    Counting pixels (not averaging) keeps one changed word visible: two boilerplate classes
    differing only in their name must not look like the same screen. A small compact change
    (mouse pointer, caret) is ignored.
    """
    if a.shape != b.shape:
        return 1e9
    ys, xs = np.nonzero(np.abs(a - b) > 24)
    n = len(ys)
    if n and n < 80 and (ys.max() - ys.min()) < 40 and (xs.max() - xs.min()) < 40:
        return 0.0
    return float(n)


def keyframes(path: str, sample_fps: float = 10.0, still: float = 8.0,
              moving_every: float = 0.5, start: float = 0.0) -> Iterator[Keyframe]:
    """Yield frames to OCR.

    still        changed-pixel count (see `_diff`) below which the screen counts as resting
    moving_every seconds between reads while the screen never rests
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Decide each sample one step late: a sample is "settled" when the *next* sample
    # looks the same, i.e. the screen has stopped moving on it.
    pending: tuple[int, float, np.ndarray, np.ndarray] | None = None
    last_emit = None        # signature of the last emitted frame
    last_emit_t = -1e9
    while True:
        if idx % step:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, bgr = cap.read()
        if not ok:
            break
        sig = _sig(bgr)
        if pending is not None:
            p_idx, p_t, p_bgr, p_sig = pending
            settled = _diff(p_sig, sig) < still
            fresh = last_emit is None or _diff(p_sig, last_emit) >= still
            if fresh and (settled or p_t - last_emit_t >= moving_every):
                last_emit, last_emit_t = p_sig, p_t
                yield Keyframe(p_idx, p_t, p_bgr, not settled)
        pending = (idx, idx / fps, bgr, sig)
        idx += 1
    if pending is not None and (last_emit is None or _diff(pending[3], last_emit) >= still):
        yield Keyframe(pending[0], pending[1], pending[2], False)
    cap.release()
