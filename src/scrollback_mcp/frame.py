"""Turn one editor screenshot into numbered source lines.

The gutter is the backbone. Every code row is keyed by the line number printed
next to it, which gives exact positions for stitching, makes blank lines
explicit, and drops anything drawn *between* numbered rows (Code Vision
"3 个用法", author hints) for free.

Text detection misses tiny isolated tokens — single-digit line numbers, `}`,
`*/`, the `*` of a Javadoc line — so rows are not taken from the detector.
Instead each numbered row is cut out by its ink extent and sent straight to the
recogniser. Spacing and indentation come from pixel geometry: in a monospace
font a gap of k cells is k spaces.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass, field

import numpy as np

from . import ocr
from .ocr import Box

FILE_EXTS = ("java|xml|yml|yaml|properties|sql|json|js|jsx|ts|tsx|vue|py|md|txt|html|css|scss|"
             "kt|kts|gradle|sh|bat|conf|ini|toml|go|rs|c|cc|cpp|h|hpp|cs|php|rb|ftl|jsp")
FILE_RE = re.compile(rf"([A-Za-z0-9_][\w\-.]*\.(?:{FILE_EXTS}))\b")
CJK_RE = re.compile(r"[　-〿㐀-鿿＀-￯]")
_NUM = re.compile(r"\d{1,5}")
_DIGIT_FIX = str.maketrans({"l": "1", "I": "1", "|": "1", "i": "1", "O": "0", "o": "0",
                            "D": "0", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9"})

INK = 40          # min channel difference from background that counts as ink
BG_NEAR = 14      # max difference for a pixel to count as background


@dataclass
class Seg:
    x0: float         # relative to the gutter's right edge
    x1: float
    text: str
    score: float
    fit: float = 0.0  # measured minus expected width, in cells (0 = the text explains the ink)
    tail_gap: float | None = None   # px between the last blob of ink and the one before it


@dataclass
class Row:
    n: int
    y: float                      # row centre in the frame
    segs: list[Seg] = field(default_factory=list)

    def text(self, char_w: float, origin: float | None = None) -> tuple[int, str]:
        """(indent columns, text) — segments left of `origin` (gutter icons) are dropped."""
        segs = [s for s in self.segs if origin is None or s.x1 > origin - 0.3 * char_w]
        if not segs:
            return 0, ""
        base = origin if origin is not None else segs[0].x0
        indent = max(0, math.floor((segs[0].x0 - base) / char_w + 0.5))
        segs = [Seg(s.x0, s.x1, s.text, s.score, s.fit, s.tail_gap) for s in segs]
        gaps = [_join(a, b, char_w) for a, b in zip(segs, segs[1:])]   # may retouch a.text
        return indent, segs[0].text + "".join(g + b.text for g, b in zip(gaps, segs[1:]))

    @property
    def score(self) -> float:
        return min((s.score for s in self.segs), default=1.0)


@dataclass
class FrameRead:
    idx: int
    t: float
    moving: bool
    width: int
    height: int
    gutter_right: float | None = None
    pitch: float = 0.0
    char_w: float = 0.0
    rows: list[Row] = field(default_factory=list)
    title_file: str | None = None
    title_module: str | None = None
    tabs: list[str] = field(default_factory=list)
    active_tab: str | None = None
    crumb: list[str] = field(default_factory=list)

    @property
    def name_hint(self) -> str | None:
        return self.active_tab or crumb_file(self.crumb) or self.title_file

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "FrameRead":
        d = dict(d)
        d["rows"] = [Row(n=r["n"], y=r["y"], segs=[Seg(**s) for s in r["segs"]]) for r in d["rows"]]
        return FrameRead(**d)


# Empty space a glyph leaves inside its own cell, in cells, per side. A gap between two
# pieces of ink = the bearings facing each other + k space cells.
_NARROW = set(".,:;'|!il1`I")
_MEDIUM = set("()[]{}\"jfrtJ")
# CJK fonts put these at one side of a ≈1.7-cell box, leaving a wide empty side
_FW_TRAILING = set("，。、；：！？）】》」』”’")      # ink on the left, room on the right
_FW_LEADING = set("（【《「『“‘")                    # ink on the right, room on the left
_FULLWIDTH_PUNCT = _FW_TRAILING | _FW_LEADING
_FW_TO_ASCII = {"，": ",", "；": ";", "：": ":", "！": "!", "？": "?", "（": "(", "）": ")"}


def _bearing(ch: str, side: str) -> float:
    """side: 'l' (room left of the ink) or 'r' (room right of it)."""
    if not ch:
        return 0.1
    if ch in _FW_TRAILING:
        return 1.1 if side == "r" else 0.15
    if ch in _FW_LEADING:
        return 1.1 if side == "l" else 0.15
    if CJK_RE.match(ch):
        return 0.12
    if ch in _NARROW:
        return 0.36
    if ch in _MEDIUM:
        return 0.22
    return 0.08


def _spaces(gap: float, char_w: float, left: str = "", right: str = "") -> int:
    k = math.floor(gap / char_w - _bearing(left, "r") - _bearing(right, "l") + 0.5)
    wide = (left and not left.isascii()) or (right and not right.isascii())
    # Latin pieces were split because a space was visible; CJK text sits apart without one
    return max(0 if wide else 1, k)


_PUNCT_TWINS = {"，": ",", "：": ":", "；": ";", "！": "!", "？": "?"}


def _join(a: "Seg", b: "Seg", char_w: float) -> str:
    """Text to put between two pieces of a row; may also settle a full-width/ASCII punctuation
    mark at the end of `a`.

    After Chinese text, `，` and `, ` look alike to the recogniser but not to a ruler: the
    full-width mark hugs the previous glyph and carries a wide empty right side; the ASCII
    one sits mid-cell and is followed by a real space. Compare both gaps with each template.
    """
    left, right = a.text[-1:], b.text[:1]
    fw = left if left in _PUNCT_TWINS else next((k for k, v in _PUNCT_TWINS.items() if v == left), None)
    before = a.text[-2:-1]
    gap = (b.x0 - a.x1) / char_w
    if fw and right and (before and not before.isascii() or not right.isascii()) and 0.9 <= gap <= 1.9:
        a.text = a.text[:-1] + fw          # Chinese prose: full-width mark, its room is the gap
        return ""
    return " " * _spaces(b.x0 - a.x1, char_w, left, right)


# ---------------------------------------------------------------- layout
@dataclass
class Layout:
    """Editor geometry, carried from frame to frame. The font does not change during a
    recording, so the cell width is a running median over every frame read so far."""
    gutter_right: float
    pitch: float
    char_w: float
    cw_samples: list[float] = field(default_factory=list)

    def learn_cell(self, cw: float) -> None:
        med = statistics.median(self.cw_samples) if self.cw_samples else None
        if med is None or abs(cw - med) <= med * 0.12:
            self.cw_samples = (self.cw_samples + [cw])[-60:]
            self.char_w = statistics.median(self.cw_samples)


def _gutter_from_boxes(boxes: list[Box]) -> tuple[float, float, float] | None:
    """Right-aligned column of increasing integers → (right edge, row pitch, a reference y)."""
    digits = [b for b in boxes if _NUM.fullmatch(b.text.strip())]
    if len(digits) < 3:
        return None
    tol = max(4.0, statistics.median(b.h for b in digits) * 0.35)
    digits.sort(key=lambda b: b.x1)
    groups: list[list[Box]] = [[digits[0]]]
    for b in digits[1:]:
        if b.x1 - groups[-1][-1].x1 <= tol:
            groups[-1].append(b)
        else:
            groups.append([b])
    best, best_inc = None, 0
    for g in groups:
        g = sorted(g, key=lambda b: b.cy)
        nums = [int(b.text) for b in g]
        inc = sum(1 for a, c in zip(nums, nums[1:]) if c > a)
        if inc > best_inc:
            best, best_inc = g, inc
    if best is None or best_inc < 2:
        return None
    steps = [(c.cy - a.cy) / (int(c.text) - int(a.text))
             for a, c in zip(best, best[1:]) if int(c.text) > int(a.text)]
    return statistics.median(b.x1 for b in best), statistics.median(steps), best[len(best) // 2].cy


def _char_width(boxes: list[Box], left: float) -> float:
    """Monospace cell width: slope of box width vs. character count over ASCII code boxes."""
    pts = [(len(b.text), b.x1 - b.x0) for b in boxes
           if b.x0 > left and len(b.text) >= 4 and b.text.isascii()]
    if len(pts) < 3:
        return 0.0
    n = np.array([p[0] for p in pts], dtype=float)
    w = np.array([p[1] for p in pts], dtype=float)
    if n.max() - n.min() < 3:
        return float(np.median(w / n))
    slope, icpt = np.polyfit(n, w, 1)
    resid = np.abs(w - (slope * n + icpt))
    keep = resid <= max(3.0, float(np.percentile(resid, 80)))
    if keep.sum() >= 3 and n[keep].max() - n[keep].min() >= 3:
        slope, _ = np.polyfit(n[keep], w[keep], 1)
    return float(slope)


def _runs(mask: np.ndarray, min_gap: int = 1) -> list[tuple[int, int]]:
    """[start, end) runs of True, merging runs separated by fewer than `min_gap` False cells."""
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    runs = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > min_gap:
            runs.append((start, prev + 1))
            start = i
        prev = i
    runs.append((start, prev + 1))
    return runs


def _bands(bgr: np.ndarray, lay: Layout, ref_y: float | None
           ) -> tuple[list[tuple[int, int]], int, int, tuple[int, int]]:
    """Vertical extents of the line numbers, the editor's top/bottom, and the digits' x-range."""
    h = bgr.shape[0]
    x0 = max(0, int(lay.gutter_right - 6.5 * lay.char_w))
    strip = bgr[:, x0:int(lay.gutter_right + 2)].astype(np.int16)
    y = int(ref_y) if ref_y is not None else h // 2
    ref = strip[max(0, y - 40):y + 40].reshape(-1, 3)
    bg = np.median(ref, axis=0)
    near = (np.abs(strip - bg).max(axis=2) < BG_NEAR).mean(axis=1) > 0.5
    top = bot = y
    while top > 0 and near[top - 1]:
        top -= 1
    while bot < h and near[bot]:
        bot += 1
    d = np.abs(strip[top:bot] - bg).max(axis=2)
    # Line numbers are the dimmest text on screen (grey on grey, then video-compressed), so
    # the ink threshold adapts to their contrast. Columns "inked" on most rows are borders.
    keep = (d > 18).mean(axis=0) < 0.5
    d = np.where(keep, d, 0)
    hot = d[d > 18]
    thr = max(18.0, 0.5 * float(np.percentile(hot, 95))) if hot.size else INK
    ink = (d > thr).any(axis=1)
    xs = np.flatnonzero((d > thr).any(axis=0))
    xr = (x0 + int(xs[0]), x0 + int(xs[-1]) + 1) if xs.size else (x0, int(lay.gutter_right + 2))
    out = []
    for a, b in _runs(ink, min_gap=max(1, int(lay.pitch * 0.2))):
        a, b = a + top, b + top
        if a <= top + 1 or b >= bot - 1:          # clipped by the viewport edge
            continue
        if lay.pitch * 0.3 <= b - a <= lay.pitch * 1.1:
            out.append((a, b))
    return out, top, bot, xr


def _ink(band: np.ndarray) -> np.ndarray:
    bg = np.median(band.reshape(-1, 3), axis=0)
    ink = np.abs(band.astype(np.int16) - bg).max(axis=2) > INK
    # pixel rows inked across most of the width are borders / highlight edges, not glyphs
    ink[ink.mean(axis=1) > 0.6] = False
    return ink


def _row_mask(band: np.ndarray) -> np.ndarray:
    """Ink that belongs to this row.

    Neighbouring rows poke in: the bottoms of tall CJK glyphs and of Code Vision hints from
    above, the tops of CJK glyphs from below. Those blobs are short and stuck to the band's
    edge; a glyph of this row (descenders included) is neither. Underscores and commas sit
    low but stay clear of the edge.
    """
    import cv2

    ink = _ink(band)
    h = ink.shape[0]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        top, height = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
        if top == 0 and height <= h * 0.25:
            ink[labels == i] = False
    return ink


def _ink_cols(band: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-column ink flags for a row band, ignoring indent guides / right-margin lines."""
    ink = _row_mask(band) if mask is None else mask
    frac = ink.mean(axis=0)
    cols = ink.any(axis=0)
    # 1-2 px wide columns inked over (almost) the full row height are guide lines, not glyphs
    for a, b in _runs(frac > 0.85):
        if b - a <= 2:
            cols[a:b] = False
    return cols


def _crop(bgr: np.ndarray, y0: int, y1: int, x0: int, x1: int, pad: int = 4) -> np.ndarray:
    import cv2

    c = bgr[max(0, y0):y1, max(0, x0 - pad):x1 + pad]
    return cv2.copyMakeBorder(c, pad, pad, 0, 0, cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- chrome (title, tabs, breadcrumb)
def crumb_file(crumb: list[str]) -> str | None:
    for seg in reversed(crumb):
        m = FILE_RE.fullmatch(seg)
        if m:
            return m.group(1)
    return None


def _tab_score(bgr: np.ndarray, b: Box) -> float:
    """Active tab = different background and/or a coloured bar under it."""
    h, w = bgr.shape[:2]
    x0, x1 = max(0, int(b.x0) - 8), min(w, int(b.x1) + 8)
    patch = bgr[max(0, int(b.y0) - 6):min(h, int(b.y1) + 6), x0:x1].astype(np.int16)
    under = bgr[min(h - 1, int(b.y1) + 2):min(h, int(b.y1) + 16), x0:x1].astype(np.int16)
    level = float(np.median(patch.mean(axis=2))) if patch.size else 0.0
    sat = float(np.percentile(under.max(axis=2) - under.min(axis=2), 90)) if under.size else 0.0
    return level + sat * 0.5


def _chrome(fr: FrameRead, bgr: np.ndarray, boxes: list[Box], top: float, bottom: float) -> None:
    above = [b for b in boxes if b.cy < top]
    below = [b for b in boxes if b.cy > bottom]

    title_box = None
    for b in sorted(above, key=lambda b: b.cy):
        if b.cy > fr.height * 0.06:
            break
        parts = re.split(r"\s*[–—]\s*|\s+-\s+", b.text)
        m = FILE_RE.search(parts[-1]) if len(parts) > 1 else None
        if m:
            title_box = b
            fr.title_file = m.group(1)
            mod = re.search(r"\[([\w\-.]+)\]", b.text)
            fr.title_module = mod.group(1) if mod else None
            break

    cands = [b for b in above if b is not title_box
             and FILE_RE.fullmatch(re.sub(r"[\s×✕]+$", "", b.text.strip()))]
    # tabs share one row: cluster by y, keep the biggest row (ties → the one nearest the editor)
    rows: list[list[Box]] = []
    for b in sorted(cands, key=lambda b: b.cy):
        if rows and abs(b.cy - rows[-1][-1].cy) < b.h * 0.6:
            rows[-1].append(b)
        else:
            rows.append([b])
    tabs = max(rows, key=lambda r: (len(r), r[0].cy)) if rows else []
    if tabs:
        tabs.sort(key=lambda b: b.x0)
        fr.tabs = [FILE_RE.search(b.text).group(1) for b in tabs]
        if len(tabs) == 1:
            fr.active_tab = fr.tabs[0]
        else:
            # the active tab is drawn lighter and/or underlined in both Darcula and Light themes
            sc = [_tab_score(bgr, b) for b in tabs]
            i = int(np.argmax(sc))
            if sc[i] - sorted(sc)[-2] > 3:
                fr.active_tab = fr.tabs[i]
    if title_box is None:
        # OCR often turns "proj – File.java" into "proj-File.java"; match against known tab names
        for b in above:
            if b.cy < fr.height * 0.06 and b not in tabs:
                hit = [t for t in fr.tabs if b.text.endswith(t)]
                if hit:
                    fr.title_file = max(hit, key=len)
                    break

    # Breadcrumb / navigation bar: "project > module > … > Name" outside the editor
    for group in (below, above):
        segs: list[str] = []
        for b in sorted(group, key=lambda b: (round(b.cy / 12), b.x0)):
            if re.search(r"[>›»]", b.text):
                segs.extend(s.strip() for s in re.split(r"[>›»]", b.text) if s.strip())
        segs = [s for s in segs if re.fullmatch(r"[\w\-.$]+", s)]
        if len(segs) >= 2:
            fr.crumb = segs
            break


# ---------------------------------------------------------------- rows
def _read_numbers(bgr: np.ndarray, bands: list[tuple[int, int]], xr: tuple[int, int]) -> list[int | None]:
    x0, x1 = xr
    res = ocr.recognize([_crop(bgr, a - 3, b + 3, x0, x1) for a, b in bands])
    out: list[int | None] = []
    for text, _ in res:
        t = text.strip().translate(_DIGIT_FIX)
        out.append(int(t) if _NUM.fullmatch(t) else None)
    return _repair_sequence(out)


def _repair_sequence(nums: list[int | None]) -> list[int | None]:
    """Line numbers of adjacent bands differ by +1 except at folds; fix lone misreads."""
    n = list(nums)
    for i in range(len(n)):
        prev = n[i - 1] if i > 0 else None
        nxt = n[i + 1] if i + 1 < len(n) else None
        if prev is not None and nxt is not None and nxt - prev == 2 and n[i] != prev + 1:
            n[i] = prev + 1
        elif n[i] is None:
            if prev is not None and (nxt is None or nxt == prev + 2):
                n[i] = prev + 1
            elif nxt is not None and prev is None:
                n[i] = nxt - 1 if nxt > 1 else None
    # anything still breaking monotonicity is unreliable
    for i in range(1, len(n)):
        if n[i] is not None and n[i - 1] is not None and n[i] <= n[i - 1]:
            n[i] = None
    return n


def _read_rows(bgr: np.ndarray, lay: Layout, bands: list[tuple[int, int]],
               nums: list[int | None], top: int, bot: int) -> list[Row]:
    """Cut every numbered row into words at visible gaps and recognise each word.

    Spaces are never taken from the recogniser (it drops them next to CJK text and invents
    them inside `<!--`); they are counted from the pixel gap between words instead.
    """
    import cv2

    w = bgr.shape[1]
    gr = lay.gutter_right
    cw = lay.char_w
    x_start = int(gr + 3)
    split = max(2, int(round(cw * 0.82)))
    rows: list[Row] = []
    clean: list[np.ndarray] = []          # per row: the band with foreign ink painted over
    pieces: list[tuple[int, int, int, float | None]] = []   # (row, x0, x1, tail gap) band coords
    for (a, b), n in zip(bands, nums):
        if n is None:
            continue
        cy = (a + b) / 2
        y0, y1 = max(top, int(cy - lay.pitch / 2)), min(bot, int(cy + lay.pitch / 2))
        rows.append(Row(n=n, y=round(cy, 1)))
        band = bgr[y0:y1, x_start:w - 4]
        raw = _ink(band)
        mask = _row_mask(band)
        band = band.copy()
        band[raw & ~mask] = np.median(band.reshape(-1, 3), axis=0).astype(band.dtype)
        clean.append(cv2.copyMakeBorder(band, 4, 4, 4, 4, cv2.BORDER_REPLICATE))
        cols = _ink_cols(band, mask)
        for c0, c1 in _runs(cols, min_gap=split):
            blobs = _runs(cols[c0:c1])
            tail = float(blobs[-1][0] - blobs[-2][1]) if len(blobs) >= 2 else None
            pieces.append((len(rows) - 1, c0, c1, tail))

    def crop(ri: int, c0: int, c1: int) -> np.ndarray:
        return clean[ri][:, c0:c1 + 8]                     # +4 px margin each side (border)

    texts = ocr.recognize([crop(ri, c0, c1) for ri, c0, c1, _ in pieces])
    texts = [(t.replace(" ", ""), sc) for t, sc in texts]
    cw_fit = _cell_from_pieces([(0, 0, 0, c0, c1) for _, c0, c1, _ in pieces], texts)
    if cw_fit:
        lay.learn_cell(cw_fit)
        cw = lay.char_w
    texts = _with_context(pieces, texts, crop, cw)
    wide_w = _wide_width([(0, 0, 0, c0, c1) for _, c0, c1, _ in pieces], texts, cw)
    for (ri, c0, c1, tail), (text, score) in zip(pieces, texts):
        if not text or (c1 - c0 < cw * 0.5 and score < 0.5 and text.isalnum()):  # specks
            continue
        text, fit = _fit_punct(text, (c1 - c0) / cw, wide_w / cw)
        x0 = x_start + c0 - gr
        rows[ri].segs.append(Seg(round(x0, 1), round(x0 + c1 - c0, 1), text, round(score, 4),
                                 round(fit, 2), tail))
    return rows


_SUSPECT = re.compile(r"[0-9/lI|.·、，,:;'`]+")


def _with_context(pieces: list, texts: list[tuple[str, float]], crop, cw: float
                  ) -> list[tuple[str, float]]:
    """Re-read doubtful short pieces together with a neighbour.

    A recogniser sees `//` alone and answers `11`; it sees `//2、` and answers `112`. With
    the next word in the same crop it reads the marker right. The neighbour's own reading
    is then cut off the combined text again.
    """
    jobs = []
    for i, ((ri, c0, c1, _), (t, sc)) in enumerate(zip(pieces, texts)):
        if c1 - c0 > cw * 3.5 or not (sc < 0.95 or _SUSPECT.fullmatch(t or "1")):
            continue
        for j, side in ((i + 1, "r"), (i - 1, "l")):
            if 0 <= j < len(pieces) and pieces[j][0] == ri and texts[j][0]:
                gap = pieces[j][1] - c1 if side == "r" else c0 - pieces[j][2]
                if gap < cw * 4:
                    jobs.append((i, j, side))
                    break
    if not jobs:
        return texts
    combo = ocr.recognize([crop(pieces[i][0], min(pieces[i][1], pieces[j][1]),
                                max(pieces[i][2], pieces[j][2])) for i, j, _ in jobs])
    out = list(texts)
    for (i, j, side), (c, sc) in zip(jobs, combo):
        c = c.replace(" ", "")
        nb = texts[j][0]
        if side == "r" and c.endswith(nb):
            mine = c[: len(c) - len(nb)]
        elif side == "l" and c.startswith(nb):
            mine = c[len(nb):]
        else:
            continue
        if mine and len(mine) <= len(texts[i][0]) + 3:
            out[i] = (mine, max(sc, texts[i][1]))
    return out


# ---------------------------------------------------------------- width check
# Full-width punctuation the recogniser confuses with ASCII. In an IDE the ASCII form is drawn
# by the monospace font (one cell) and the full-width form by the CJK fallback (≈1.7 cells),
# so the measured width tells them apart.
_CONFUSABLE = {"，": ",", "；": ";", "：": ":", "（": "(", "）": ")", "！": "!", "？": "?",
               "“": '"', "”": '"', "‘": "'", "’": "'", "［": "[", "］": "]"}


def _is_wide(ch: str) -> bool:
    return not ch.isascii()


def _expected(text: str, wide: float) -> float:
    """Ink width in cells: advances minus the empty room at both ends."""
    adv = sum(wide if _is_wide(c) else 1.0 for c in text)
    return adv - _bearing(text[0], "l") - _bearing(text[-1], "r")


def _cell_from_pieces(pieces: list, texts: list[tuple[str, float]]) -> float | None:
    """Cell width from many confidently read ASCII words: ink width ≈ cell * chars + const."""
    n, w = [], []
    for (_, _, _, x0, x1), (t, sc) in zip(pieces, texts):
        t = t.replace(" ", "")
        if len(t) >= 3 and sc > 0.97 and t.isascii() and not (set(t[0] + t[-1]) & (_NARROW | _MEDIUM)):
            n.append(len(t))
            w.append(x1 - x0)
    if len(n) < 8 or max(n) - min(n) < 4:
        return None
    slope, _ = np.polyfit(np.array(n, float), np.array(w, float), 1)
    return float(slope)


def _wide_width(pieces: list, texts: list[tuple[str, float]], cw: float) -> float:
    """Advance of a CJK glyph in px, learned from pieces that are pure CJK; else 1.7 cells."""
    ws = []
    for (_, _, _, x0, x1), (t, sc) in zip(pieces, texts):
        t = t.replace(" ", "")
        if len(t) >= 2 and sc > 0.9 and all(CJK_RE.match(c) and c not in _FULLWIDTH_PUNCT for c in t):
            ws.append((x1 - x0 + 2 * 0.12 * cw) / len(t))
    return statistics.median(ws) if ws else 1.7 * cw


def _fit_punct(text: str, width: float, wide: float) -> tuple[str, float]:
    """Pick ASCII vs full-width punctuation by how well each explains the measured width."""
    # curly quotes / full-width semicolon at a piece edge or next to ASCII are code, not prose
    t = list(text)
    for i, c in enumerate(t):
        if c in "“”‘’":
            left = t[i - 1] if i > 0 else " "
            right = t[i + 1] if i + 1 < len(t) else " "
            if left.isascii() or right.isascii() or i in (0, len(t) - 1):
                t[i] = _CONFUSABLE[c]
    if t and t[-1] == "；":
        t[-1] = ";"
    for i in range(len(t) - 1):
        # its empty right side would have split the piece here; so it is the ASCII twin
        if t[i] in _FW_TRAILING and t[i] in _FW_TO_ASCII:
            t[i] = _FW_TO_ASCII[t[i]]
    text = "".join(t)
    best = (abs(width - _expected(text, wide)), text)
    conf = [i for i, c in enumerate(text) if c in _CONFUSABLE or c in _CONFUSABLE.values()]
    if conf and len(conf) <= 6:
        # try flipping each confusable on its own, keep flips that bring the width closer
        cur = list(text)
        for i in conf:
            c = cur[i]
            alt = _CONFUSABLE.get(c) or next((k for k, v in _CONFUSABLE.items()
                                              if v == c and k not in "“”‘’"), None)
            if alt is None:
                continue
            near_cjk = any(0 <= j < len(cur) and CJK_RE.match(cur[j]) for j in (i - 1, i + 1))
            if alt in _FULLWIDTH_PUNCT and not near_cjk:
                continue
            trial = cur.copy()
            trial[i] = alt
            err = abs(width - _expected("".join(trial), wide))
            if err + 0.25 < best[0]:
                cur = trial
                best = (err, "".join(trial))
    text = best[1]
    return text, width - _expected(text, wide)


# ---------------------------------------------------------------- main entry
def parse(bgr: np.ndarray, boxes: list[Box], idx: int, t: float, moving: bool,
          memory: Layout | None = None) -> tuple[FrameRead, Layout | None]:
    """Read one frame. `memory` is the layout learned from earlier frames (short files
    show no two-digit line numbers, so the detector alone cannot find their gutter)."""
    h, w = bgr.shape[:2]
    fr = FrameRead(idx=idx, t=round(t, 3), moving=moving, width=w, height=h)
    g = _gutter_from_boxes(boxes)
    lay, ref_y = None, None
    if g:
        right, pitch, ref_y = g
        cw = (memory.char_w if memory and memory.cw_samples else 0.0) or _char_width(boxes, right)             or (memory.char_w if memory else 0.0)
        if cw > 0:
            lay = Layout(right, pitch, cw, list(memory.cw_samples) if memory else [])
    elif memory:
        lay = memory
    if lay is None:
        _chrome(fr, bgr, boxes, top=h * 0.12, bottom=h * 0.92)
        return fr, memory

    if ref_y is None:
        code = [b for b in boxes if b.x0 > lay.gutter_right]
        ref_y = statistics.median(b.cy for b in code) if code else h / 2
    bands, top, bot, xr = _bands(bgr, lay, ref_y)
    if len(bands) < 1:
        _chrome(fr, bgr, boxes, top=h * 0.12, bottom=h * 0.92)
        return fr, memory
    fr.gutter_right, fr.pitch, fr.char_w = lay.gutter_right, lay.pitch, lay.char_w
    nums = _read_numbers(bgr, bands, xr)
    fr.rows = _read_rows(bgr, lay, bands, nums, top, bot)
    _chrome(fr, bgr, boxes, top=top, bottom=bot)
    return fr, lay
