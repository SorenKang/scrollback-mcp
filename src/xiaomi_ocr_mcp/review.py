"""Review queue: which line ranges deserve a second look, the frame crop that shows
them, and applying fixes that survive a rebuild."""

from __future__ import annotations

import json
from pathlib import Path

from .rebuild import _load_reads, load_overrides, save_overrides, work_dir

MAX_RANGE = 15


def _state(out: Path) -> dict:
    return json.loads((work_dir(out) / "state.json").read_text(encoding="utf-8"))


def _verify(out: Path) -> dict:
    p = work_dir(out) / "verify.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def queue(out: str | Path, state: dict | None = None) -> list[dict]:
    """Ranges to review, most urgent first: syntax errors, never-seen lines, then OCR doubts."""
    out = Path(out)
    state = state or _state(out)
    syntax = _verify(out).get("by_file", {})
    items = []
    for f in state["files"]:
        errs = {e["line"]: e["msg"] for e in syntax.get(f["path"], [])}
        bad = []
        for ln in f["lines"]:
            flags = [x for x in ln["flags"] if x != "reviewed"]
            if ln["n"] in errs:
                flags.append("syntax")
            if flags:
                bad.append((ln["n"], flags))
        ranges: list[list] = []
        for n, flags in bad:
            if ranges and n - ranges[-1][1] <= 2 and n - ranges[-1][0] < MAX_RANGE:
                ranges[-1][1] = n
                ranges[-1][2].update(flags)
            else:
                ranges.append([n, n, set(flags)])
        for a, b, flags in ranges:
            pri = 0 if "syntax" in flags else 1 if "missing" in flags else 2
            item = {"path": f["path"], "start": a, "end": b, "flags": sorted(flags), "priority": pri}
            msgs = [f"L{n}: {m}" for n, m in errs.items() if a <= n <= b]
            if msgs:
                item["errors"] = msgs
            items.append(item)
    items.sort(key=lambda i: (i["priority"], i["path"], i["start"]))
    return items


def write_report(out: str | Path, state: dict | None = None) -> dict:
    out = Path(out)
    state = state or _state(out)
    q = queue(out, state)
    ver = _verify(out)
    lines = ["# Rebuild report", ""]
    if state.get("video"):
        lines.append(f"video: `{state['video']}`")
    total = sum(len(f["lines"]) for f in state["files"])
    flagged = sum(i["end"] - i["start"] + 1 for i in q)
    missing = sum(1 for f in state["files"] for ln in f["lines"] if "missing" in ln["flags"])
    lines += [f"files: {len(state['files'])}, lines: {total}, lines to review: {flagged}, "
              f"never seen: {missing}", ""]
    if ver:
        lines.append(f"syntax check: {ver.get('errors', 0)} errors in "
                     f"{ver.get('files_with_errors', 0)} files" + ("" if ver.get("javac", True)
                                                                    else " (javac not found)"))
        lines.append("")
    lines += ["| file | lines | to review | notes |", "|---|---|---|---|"]
    for f in state["files"]:
        n_rev = sum(i["end"] - i["start"] + 1 for i in q if i["path"] == f["path"])
        notes = "; ".join(f["notes"]) + ("" if f["path_sure"] else " (path guessed)")
        lines.append(f"| `{f['path']}` | {len(f['lines'])} | {n_rev} | {notes} |")
    lines += ["", "## Review queue", "",
              "`xiaomi-ocr-mcp crop <out> <file> <start> <end>` shows the frame; "
              "`xiaomi-ocr-mcp set <out> <file> <line> <text>` fixes it.", ""]
    for i in q:
        err = f" — {'; '.join(i['errors'])}" if i.get("errors") else ""
        lines.append(f"- `{i['path']}` L{i['start']}-{i['end']} [{', '.join(i['flags'])}]{err}")
    (work_dir(out) / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"flagged": flagged, "missing": missing, "ranges": len(q)}


# ---------------------------------------------------------------- crops
def crop(out: str | Path, path: str, start: int, end: int | None = None,
         context: int = 1) -> dict:
    """Cut the region of the best frame showing lines start..end. Returns PNG path + current text."""
    import cv2

    from . import video

    out = Path(out)
    end = end or start
    state = _state(out)
    f = next((x for x in state["files"] if x["path"] == path), None)
    if f is None:
        raise KeyError(f"no such file in the rebuild: {path}")
    reads = _load_reads(work_dir(out) / "reads.jsonl")
    by_idx = {r.idx: r for r in reads}
    want = set(range(start, end + 1))
    best, best_key = None, None
    for idx in f["frames"]:
        r = by_idx.get(idx)
        if r is None:
            continue
        have = {row.n for row in r.rows} & want
        if not have:
            continue
        key = (len(have), not r.moving, -abs(_mid_y(r, have) - r.height / 2))
        if best_key is None or key > best_key:
            best, best_key = r, key
    current = _current_lines(out, path, start, end)
    if best is None:
        return {"ok": False, "reason": "these lines never appeared on screen; type them in with set_lines",
                "current": current}
    bgr = video.frame_at(state["video"], best.idx)
    ys = [row.y for row in best.rows if start - context <= row.n <= end + context]
    y0 = int(max(0, min(ys) - best.pitch))
    y1 = int(min(best.height, max(ys) + best.pitch))
    x0 = int(max(0, (best.gutter_right or 0) - 70))
    img = bgr[y0:y1, x0:]
    # trim empty background on the right so the crop stays readable
    bg = img[:, -1:].astype(int)
    used = (abs(img.astype(int) - bg).max(axis=2) > 40).any(axis=0).nonzero()[0]
    if used.size:
        img = img[:, :min(img.shape[1], used[-1] + 40)]
    shots = work_dir(out) / "crops"
    shots.mkdir(exist_ok=True)
    png = shots / f"{path.replace('/', '_')}_L{start}-{end}_f{best.idx}.png"
    cv2.imencode(".png", img)[1].tofile(str(png))
    covered = sorted({row.n for row in best.rows} & want)
    return {"ok": True, "image": str(png), "frame": best.idx, "t": best.t,
            "lines_in_crop": [covered[0], covered[-1]] if covered else None, "current": current}


def _mid_y(r, ns: set[int]) -> float:
    ys = [row.y for row in r.rows if row.n in ns]
    return sum(ys) / len(ys) if ys else 0.0


def _current_lines(out: Path, path: str, start: int, end: int) -> list[str]:
    p = out / path
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").split("\n")
    return [f"{n:>4}| {lines[n - 1]}" for n in range(start, min(end, len(lines)) + 1)]


# ---------------------------------------------------------------- fixes
def set_lines(out: str | Path, path: str, start: int, text: str) -> dict:
    """Replace lines start.. with `text` (one entry per line, indentation included).

    Line numbers are fixed by the gutter, so this replaces rather than inserts. The fix is
    stored in overrides.json and survives rebuilds.
    """
    out = Path(out)
    new = text.split("\n")
    ov = load_overrides(out)
    file_ov = ov.setdefault(path, {})
    for i, ln in enumerate(new):
        file_ov[str(start + i)] = ln.rstrip()
    save_overrides(out, ov)

    p = out / path
    lines = p.read_text(encoding="utf-8").split("\n") if p.exists() else []
    if lines and lines[-1] == "":
        lines.pop()
    while len(lines) < start - 1 + len(new):
        lines.append("")
    lines[start - 1:start - 1 + len(new)] = [ln.rstrip() for ln in new]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    sp = work_dir(out) / "state.json"
    if sp.exists():
        state = json.loads(sp.read_text(encoding="utf-8"))
        for f in state["files"]:
            if f["path"] == path:
                for ln in f["lines"]:
                    if start <= ln["n"] < start + len(new):
                        ln["flags"], ln["conf"] = ["reviewed"], 1.0
        sp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "path": path, "lines": [start, start + len(new) - 1]}
