"""Render a fake IntelliJ IDEA screen recording of a real source tree.

Used as ground truth for the pipeline: every pixel comes from known files, so the
reconstruction can be diffed line by line against the originals.

    python tools/make_video.py <project_dir> -o demo.mp4 [--files 10] [--code-vision]

What it imitates (the parts that matter for OCR):
  * Darcula colours, Consolas code font, Microsoft YaHei fallback for CJK
  * gutter with right-aligned line numbers
  * Code Vision rows ("3 个用法  zss") between lines — rows without a line number
  * tab strip with an active tab, project tree on the left, breadcrumb status bar
  * page-by-page scrolling with an animated transition, plus some wheel scrolling
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30
BG = (43, 43, 43)
PANEL = (60, 63, 65)
GUTTER_FG = (96, 99, 102)
FG = (169, 183, 198)
KW = (204, 120, 50)
STR = (106, 135, 89)
COMMENT = (128, 128, 128)
ANNOT = (187, 181, 41)
HINT = (120, 120, 120)

TREE_W = 300
GUTTER_RIGHT = 352
CODE_X = 372
TOP = 80             # below title bar + tab strip
BOTTOM = 1046        # above status bar
LINE_H = 22

MONO = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 16)
MONO_SMALL = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 13)
CJK = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
UI = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
CELL = MONO.getlength("M")

KEYWORDS = set("""abstract assert boolean break byte case catch char class const continue default do
double else enum extends final finally float for goto if implements import instanceof int interface
long native new package private protected public return short static strictfp super switch
synchronized this throw throws transient try void volatile while true false null var record""".split())
TOKEN = re.compile(r'(//.*$|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|@\w+|\w+|\s+|.)')


def _draw_text(d: ImageDraw.ImageDraw, x: float, y: int, text: str, fill) -> float:
    """Mono font for ASCII, YaHei for everything else — same fallback IDEA uses."""
    for ch in text:
        if ord(ch) < 128:
            d.text((x, y), ch, font=MONO, fill=fill)
            x += CELL
        else:
            d.text((x, y + 1), ch, font=CJK, fill=fill)
            x += CJK.getlength(ch)
    return x


def _spans(line: str, in_block: bool) -> tuple[list[tuple[str, tuple]], bool]:
    """Very small Java/XML highlighter. Returns coloured spans and block-comment state."""
    out: list[tuple[str, tuple]] = []
    i = 0
    if in_block:
        end = line.find("*/")
        if end < 0:
            return [(line, COMMENT)], True
        out.append((line[:end + 2], COMMENT))
        i = end + 2
    rest = line[i:]
    for m in TOKEN.finditer(rest):
        tok = m.group(0)
        if tok.startswith("/*") and not tok.endswith("*/"):
            out.append((rest[m.start():], COMMENT))
            return out, True
        if tok.startswith(("//", "/*")):
            col = COMMENT
        elif tok.startswith(('"', "'")):
            col = STR
        elif tok.startswith("@"):
            col = ANNOT
        elif tok in KEYWORDS:
            col = KW
        else:
            col = FG
        out.append((tok, col))
    return out, False


_DECL = re.compile(r"^\s*(public|private|protected)\b.*(\(|\bclass\b|\binterface\b|\benum\b)")


def render_file(text: str, code_vision: bool, rng: random.Random) -> tuple[Image.Image, list[int]]:
    """Render the whole file as one tall image (gutter + code). Returns image and row y of each line."""
    lines = text.expandtabs(4).split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    rows: list[tuple[str, int | None]] = []          # (text, line number or None for hint rows)
    for n, ln in enumerate(lines, 1):
        if code_vision and _DECL.match(ln) and not ln.rstrip().endswith(";"):
            indent = len(ln) - len(ln.lstrip())
            hint = f"{rng.randint(0, 9)} 个用法" + rng.choice(["", "  zss", "  zss +1", "  zqq"])
            rows.append((" " * indent + hint, None))
        rows.append((ln, n))
    img = Image.new("RGB", (W - TREE_W, len(rows) * LINE_H + BOTTOM - TOP), BG)
    d = ImageDraw.Draw(img)
    ys: list[int] = []
    in_block = False
    for r, (ln, n) in enumerate(rows):
        y = r * LINE_H + 2
        if n is None:
            indent = len(ln) - len(ln.lstrip())
            x = CODE_X - TREE_W + indent * CELL
            for ch in ln.strip():
                f = UI if ord(ch) > 127 else MONO_SMALL
                d.text((x, y + 3), ch, font=f, fill=HINT)
                x += f.getlength(ch)
            continue
        ys.append(y)
        num = str(n)
        d.text((GUTTER_RIGHT - TREE_W - MONO.getlength(num), y), num, font=MONO, fill=GUTTER_FG)
        x = CODE_X - TREE_W
        spans, in_block = _spans(ln, in_block)
        for tok, col in spans:
            x = _draw_text(d, x, y, tok, col)
    return img, ys


def chrome(project: str, tabs: list[str], active: str, crumb: str, tree: list[str]) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 40), fill=PANEL)
    d.text((12, 10), f"{project} – {active}", font=UI, fill=(187, 187, 187))
    d.rectangle((TREE_W, 40, W, TOP - 2), fill=(49, 51, 53))
    x = TREE_W + 8
    for t in tabs:
        w = UI.getlength(t) + 34
        if t == active:
            d.rectangle((x, 42, x + w, TOP - 3), fill=(78, 82, 84))
            d.rectangle((x, TOP - 5, x + w, TOP - 3), fill=(74, 136, 199))
        d.text((x + 18, 50), t, font=UI, fill=(187, 187, 187))
        x += w + 2
    d.rectangle((0, 40, TREE_W - 2, BOTTOM), fill=PANEL)
    for i, t in enumerate(tree[:40]):
        d.text((10, 50 + i * 24), t, font=UI, fill=(187, 187, 187))
    d.rectangle((0, BOTTOM, W, H), fill=PANEL)
    d.text((10, BOTTOM + 8), crumb, font=UI, fill=(187, 187, 187))
    d.text((W - 300, BOTTOM + 8), "12:5  CRLF  UTF-8  4 spaces", font=UI, fill=(187, 187, 187))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("-o", "--out", default="demo.mp4")
    ap.add_argument("--files", type=int, default=0, help="limit to N files (0 = all)")
    ap.add_argument("--code-vision", action="store_true", help="insert '个用法' hint rows")
    ap.add_argument("--wheel", type=float, default=0.3, help="share of pages scrolled by wheel")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ext", default=".java,.xml,.yml")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.project)
    exts = tuple(args.ext.split(","))
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in exts
                   and ".idea" not in p.parts and "target" not in p.parts)
    if args.files:
        files = files[: args.files]
    tree = sorted({str(p.parent.relative_to(root)).replace("\\", "/").split("/java/")[-1]
                   for p in files})

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, FPS, (W, H))
    view_h = BOTTOM - TOP
    tabs: list[str] = []
    n_frames = 0

    def emit(frame: Image.Image, count: int) -> None:
        nonlocal n_frames
        bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        for _ in range(count):
            vw.write(bgr)
        n_frames += count

    manifest = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        text = p.read_text(encoding="utf-8", errors="replace")
        tall, ys = render_file(text, args.code_vision, rng)
        if p.name in tabs:
            tabs.remove(p.name)
        tabs.insert(0, p.name)
        tabs = tabs[:6]
        # IDEA's navigation bar: project > module > … > Class (no .java), other files keep their name
        segs = ["sky-take-out"] + rel.split("/")
        if p.suffix == ".java":
            segs[-1] = p.stem
        base = chrome("sky-take-out", tabs, p.name, " > ".join(segs), tree)
        manifest.append(rel)

        def frame_at(off: int) -> Image.Image:
            f = base.copy()
            f.paste(tall.crop((0, off, W - TREE_W, off + view_h)), (TREE_W, TOP))
            return f

        max_off = max(0, len(ys) and ys[-1] + LINE_H * 3 - view_h + 40)
        off = 0
        emit(frame_at(0), int(FPS * 1.0))
        while off < max_off:
            target = min(max_off, off + view_h - 3 * LINE_H)
            if rng.random() < args.wheel:          # wheel: many small steps, never fully still
                while off < target:
                    off = min(target, off + LINE_H * 3)
                    emit(frame_at(off), 3)
            else:                                  # page down: short animation
                start = off
                for k in range(1, 7):
                    off = int(start + (target - start) * k / 6)
                    emit(frame_at(off), 1)
            emit(frame_at(off), int(FPS * rng.uniform(0.5, 1.0)))
    vw.release()
    Path(args.out).with_suffix(".files.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(files)} files, {n_frames} frames, {n_frames / FPS:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
