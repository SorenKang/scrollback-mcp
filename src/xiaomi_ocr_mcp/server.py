"""MCP server: rebuild a project from a screen recording of an IDE.

Typical session for an agent:
    video_info → rebuild_start → rebuild_status (until done) → review_queue
    → for each item: view_lines (look at the image) → set_lines if the text is wrong
    → verify → repeat review_queue until it is empty

stdio transport; logs go to stderr only.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="[xiaomi-ocr-mcp] %(levelname)s %(message)s")

try:  # mcp >= 2
    from mcp.server.mcpserver import Image, MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server, Image  # type: ignore

mcp = _Server("xiaomi-ocr-mcp")

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _key(out: str) -> str:
    return str(Path(out).resolve())


@mcp.tool()
def video_info(video: str, samples: int = 3) -> dict[str, Any]:
    """Duration, fps and resolution of a recording, plus a few frames saved as PNG so you can
    check that the editor (line numbers, tabs, navigation bar) is readable before rebuilding."""
    import cv2

    from . import video as V

    info = V.info(video)
    out = Path(video).with_suffix("")
    paths = []
    for i in range(samples):
        t = info["seconds"] * (i + 1) / (samples + 1)
        p = Path(f"{out}.sample{i + 1}.png")
        cv2.imencode(".png", V.frame_at(video, int(t * info["fps"])))[1].tofile(str(p))
        paths.append({"t": round(t, 1), "path": str(p)})
    return {**info, "samples": paths}


@mcp.tool()
def rebuild_start(video: str, out: str, sample_fps: float = 10.0) -> dict[str, Any]:
    """Start rebuilding the project shown in `video` into directory `out` (runs in the
    background; poll rebuild_status). Re-running on the same `out` resumes from its cache
    and keeps every fix made with set_lines."""
    from . import rebuild

    k = _key(out)
    with _jobs_lock:
        job = _jobs.get(k)
        if job and job["state"] == "running":
            return {"ok": False, "reason": "already running", "progress": job["progress"]}
        job = {"state": "running", "progress": {}, "result": None, "error": None,
               "started": time.time()}
        _jobs[k] = job

    def work() -> None:
        try:
            job["result"] = rebuild.run(video, out, sample_fps=sample_fps,
                                        progress=lambda p: job.__setitem__("progress", p))
            if job["result"].get("verify"):
                job["result"]["verify"].pop("by_file", None)
            job["state"] = "done"
        except Exception as exc:  # noqa: BLE001
            job["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
            job["state"] = "failed"

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "out": k, "hint": "poll rebuild_status(out)"}


@mcp.tool()
def rebuild_status(out: str) -> dict[str, Any]:
    """Progress of a rebuild started with rebuild_start: state running|done|failed."""
    job = _jobs.get(_key(out))
    if not job:
        return {"state": "unknown", "hint": "no rebuild started in this server session"}
    return {"state": job["state"], "progress": job["progress"], "result": job["result"],
            "error": job["error"], "elapsed": round(time.time() - job["started"], 1)}


@mcp.tool()
def review_queue(out: str, limit: int = 50) -> dict[str, Any]:
    """Line ranges that need a look, most urgent first (syntax errors, lines never shown,
    OCR doubts). Each item: path, start, end, flags. Use view_lines on each."""
    from . import review

    q = review.queue(out)
    report = (Path(out) / ".xocr" / "REPORT.md").read_text(encoding="utf-8")
    summary = "\n".join(report.split("## Review queue")[0].splitlines()[:8])
    return {"total": len(q), "items": q[:limit], "summary": summary}


@mcp.tool()
def view_lines(out: str, path: str, start: int, end: int | None = None):  # image + text, unstructured
    """Show the recorded frame for lines start..end of a rebuilt file, next to the current
    text. Compare them; if the text is wrong, call set_lines."""
    from . import review

    res = review.crop(out, path, start, end)
    text = "current text:\n" + "\n".join(res.get("current", []))
    if not res.get("ok"):
        return [res["reason"] + "\n" + text]
    return [Image(path=res["image"]),
            f"frame {res['frame']} (t={res['t']}s), lines {res['lines_in_crop']}\n{text}"]


@mcp.tool()
def set_lines(out: str, path: str, start: int, text: str) -> dict[str, Any]:
    """Replace lines of a rebuilt file, starting at line `start`; `text` holds one line per
    '\\n' with its indentation. Line numbers come from the editor gutter, so this replaces
    and never shifts lines. Fixes survive later rebuilds."""
    from . import review

    return review.set_lines(out, path, start, text)


@mcp.tool()
def read_file(out: str, path: str, start: int = 1, end: int | None = None) -> str:
    """Numbered lines of a rebuilt file (for clients without file access)."""
    lines = (Path(out) / path).read_text(encoding="utf-8").split("\n")
    end = end or len(lines)
    return "\n".join(f"{n:>4}| {lines[n - 1]}" for n in range(start, min(end, len(lines)) + 1))


@mcp.tool()
def verify(out: str, build: bool = False) -> dict[str, Any]:
    """Syntax-check the rebuilt tree (javac parse phase, XML, YAML). build=true also runs
    `mvn compile` (needs network for dependencies). Refreshes the review queue."""
    from . import review
    from . import verify as V

    res = V.check(out, build=build)
    review.write_report(out)
    return res


@mcp.tool()
def ocr_image(image_path: str) -> dict[str, Any]:
    """Read code from a screenshot of an editor (line numbers in the gutter help a lot)."""
    import cv2
    import numpy as np

    from . import frame as F
    from . import ocr, stitch

    bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    boxes = ocr.read(bgr)
    fr, _ = F.parse(bgr, boxes, 0, 0.0, False)
    if fr.rows:
        cw, origin = stitch.calibrate([fr])
        lines = []
        for r in fr.rows:
            indent, t = r.text(cw, origin)
            lines.append(f"{r.n:>4}| {' ' * indent}{t}")
        return {"ok": True, "gutter": True, "text": "\n".join(lines)}
    boxes.sort(key=lambda b: (round(b.cy / max(1.0, b.h)), b.x0))
    return {"ok": True, "gutter": False, "text": "\n".join(b.text for b in boxes)}
