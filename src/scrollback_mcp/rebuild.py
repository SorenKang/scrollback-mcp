"""Video → project directory, with a resumable cache and a review queue.

Layout of the output directory:

    <out>/                      the rebuilt project (pom.xml, src/main/java/…)
    <out>/.xocr/reads.jsonl     one parsed frame per line (cache; reruns skip finished frames)
    <out>/.xocr/state.json      per-file, per-line result: text, confidence, flags, best frame
    <out>/.xocr/overrides.json  manual fixes (`set_lines`); re-applied on every rebuild
    <out>/.xocr/REPORT.md       what needs a human (or Claude) to look at
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from . import frame as F
from . import ocr, stitch, video

WORK = ".xocr"


def work_dir(out: str | Path) -> Path:
    return Path(out) / WORK


def _load_reads(path: Path) -> list[F.FrameRead]:
    if not path.exists():
        return []
    reads = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            reads.append(F.FrameRead.from_json(json.loads(line)))
    return reads


def load_overrides(out: str | Path) -> dict[str, dict[str, str]]:
    p = work_dir(out) / "overrides.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_overrides(out: str | Path, ov: dict) -> None:
    p = work_dir(out) / "overrides.json"
    p.write_text(json.dumps(ov, ensure_ascii=False, indent=1), encoding="utf-8")


def read_video(video_path: str, out: str | Path, sample_fps: float = 10.0,
               progress: Callable[[dict], None] | None = None) -> list[F.FrameRead]:
    """OCR every keyframe (cached in reads.jsonl; an interrupted run resumes where it stopped)."""
    work = work_dir(out)
    work.mkdir(parents=True, exist_ok=True)
    meta_p = work / "meta.json"
    vp = Path(video_path).resolve()
    meta = {"video": str(vp), "size": vp.stat().st_size, "sample_fps": sample_fps}
    cache = work / "reads.jsonl"
    if not meta_p.exists() or json.loads(meta_p.read_text(encoding="utf-8")) != meta:
        cache.unlink(missing_ok=True)
        meta_p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    reads = _load_reads(cache)
    info = video.info(str(vp))
    done_idx = reads[-1].idx if reads else -1
    start = reads[-1].t if reads else 0.0

    memory: F.Layout | None = None
    for r in reversed(reads):
        if r.gutter_right is not None:
            memory = F.Layout(r.gutter_right, r.pitch, r.char_w)
            break

    t0 = time.time()
    with cache.open("a", encoding="utf-8") as fh:
        for kf in video.keyframes(str(vp), sample_fps=sample_fps, start=start):
            if kf.idx <= done_idx:
                continue
            boxes = ocr.read(kf.bgr)
            fr, memory = F.parse(kf.bgr, boxes, kf.idx, kf.t, kf.moving, memory)
            reads.append(fr)
            fh.write(json.dumps(fr.to_json(), ensure_ascii=False) + "\n")
            fh.flush()
            if progress:
                progress({"stage": "ocr", "t": round(kf.t, 1), "seconds": info["seconds"],
                          "frames": len(reads), "elapsed": round(time.time() - t0, 1)})
    return reads


def assemble(out: str | Path, reads: list[F.FrameRead] | None = None) -> dict:
    """Stitch cached reads into files, apply overrides, write the tree + state + report."""
    out = Path(out)
    work = work_dir(out)
    if reads is None:
        reads = _load_reads(work / "reads.jsonl")
    docs, calib = stitch.build(reads)
    overrides = load_overrides(out)

    written = []
    for d in docs:
        for n_str, text in overrides.get(d.path, {}).items():
            n = int(n_str)
            while len(d.lines) < n:
                d.lines.append(stitch.Line(n=len(d.lines) + 1, flags=["missing"]))
            ln = d.lines[n - 1]
            stripped = text.lstrip(" ")
            ln.indent, ln.text = len(text) - len(stripped), stripped
            ln.flags, ln.conf = ["reviewed"], 1.0
        p = out / d.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(d.text, encoding="utf-8", newline="\n")
        written.append(d.path)

    state = {
        "video": json.loads((work / "meta.json").read_text(encoding="utf-8")).get("video")
        if (work / "meta.json").exists() else None,
        "calibration": calib,
        "files": [{
            "path": d.path, "path_sure": d.path_sure, "notes": d.notes,
            "frames": [reads[i].idx for i in d.frames],
            "lines": [{**asdict(ln), "frame": reads[ln.frame].idx if ln.frame is not None else None,
                       "y": _row_y(reads, ln.frame, ln.n)} for ln in d.lines],
        } for d in docs],
    }
    (work / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    from . import review

    report = review.write_report(out, state)
    return {"files": len(written), "lines": sum(len(d.lines) for d in docs),
            "flagged": report["flagged"], "missing": report["missing"],
            "report": str(work / "REPORT.md")}


def _row_y(reads: list[F.FrameRead], fi: int | None, n: int) -> float | None:
    if fi is None:
        return None
    for r in reads[fi].rows:
        if r.n == n:
            return r.y
    return None


def run(video_path: str, out: str | Path, sample_fps: float = 10.0,
        progress: Callable[[dict], None] | None = None, verify: bool = True) -> dict:
    t0 = time.time()
    reads = read_video(video_path, out, sample_fps=sample_fps, progress=progress)
    if progress:
        progress({"stage": "assemble", "frames": len(reads)})
    res = assemble(out, reads)
    if verify:
        from . import verify as V

        if progress:
            progress({"stage": "verify"})
        res["verify"] = V.check(out)
        from . import review

        review.write_report(out)        # re-render with syntax errors folded in
    res["frames"] = len(reads)
    res["seconds"] = round(time.time() - t0, 1)
    res["backend"] = ocr.backend()
    return res
