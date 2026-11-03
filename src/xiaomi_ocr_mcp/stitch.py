"""Frames → files.

1. Calibrate: monospace cell width and the x of column 0, over the whole video.
2. Segment: each frame joins the document it agrees with (same file name hint,
   same text on shared line numbers), or starts a new one.
3. Vote: every line number keeps all its observations; the most supported text wins.
4. Name: Java files are named by their public type and placed by their package;
   the navigation bar supplies module directories and non-Java paths.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

from .clean import fix_file, fix_line
from .frame import CJK_RE, FILE_RE, FrameRead


def norm(s: str) -> str:
    return "".join(s.split())


@dataclass
class Obs:
    frame: int            # index into the reads list
    text: str
    indent: int
    score: float
    moving: bool
    fit: float = 0.0      # worst width residual of the row's pieces, in cells


@dataclass
class Line:
    n: int
    text: str = ""
    indent: int = 0
    conf: float = 0.0
    votes: int = 0
    flags: list[str] = field(default_factory=list)
    frame: int | None = None          # best frame showing this line


@dataclass
class Doc:
    id: int
    key: str | None
    crumb: tuple[str, ...] = ()
    obs: dict[int, list[Obs]] = field(default_factory=dict)
    frames: list[int] = field(default_factory=list)
    hints: Counter = field(default_factory=Counter)
    crumbs: Counter = field(default_factory=Counter)
    lines: list[Line] = field(default_factory=list)
    path: str = ""
    path_sure: bool = True
    notes: list[str] = field(default_factory=list)

    def best(self, n: int) -> str | None:
        obs = self.obs.get(n)
        if not obs:
            return None
        return Counter(norm(o.text) for o in obs).most_common(1)[0][0]

    @property
    def text(self) -> str:
        return "\n".join(" " * ln.indent + ln.text if ln.text else "" for ln in self.lines) + "\n"


# ---------------------------------------------------------------- calibration
def calibrate(reads: list[FrameRead]) -> tuple[float, float]:
    """(cell width, x of column 0 relative to the gutter edge)."""
    cws = [r.char_w for r in reads if r.rows and r.char_w > 0]
    cw = statistics.median(cws) if cws else 8.0
    xs = sorted(r.rows[i].segs[0].x0 for r in reads for i in range(len(r.rows)) if r.rows[i].segs)
    if not xs:
        return cw, 0.0
    need = max(3, int(len(xs) * 0.01))
    j = 0
    for i, v in enumerate(xs):
        while xs[j] < v - cw * 0.5:
            j += 1
        # window (v - cw/2, v]: enough rows start here → leftmost text column
        if i - j + 1 >= need:
            return cw, statistics.median(xs[j:i + 1])
    return cw, xs[0]


# ---------------------------------------------------------------- segmentation
def _agreement(doc: Doc, rows: dict[int, str]) -> tuple[int, int]:
    """(overlapping line numbers, how many of them agree). A clash on a declaration line
    (package, class name) means a different file, whatever the rest says — boilerplate
    classes differ in nothing else."""
    ov = ag = 0
    for n, t in rows.items():
        b = doc.best(n)
        if b is None:
            continue
        ov += 1
        if b == t:
            ag += 1
        elif _identity(b) and _identity(t) and _identity(b) != _identity(t):
            return ov, -1
        elif b and t and _similar(b, t):
            ag += 1
    return ov, ag


_IDENT_RE = re.compile(
    r"^(?:package([\w.]+)[;；]|(?:public|final|abstract|static)*"
    r"(?:class|interface|enum|record|@interface)([A-Za-z_$][\w$]*?)(?:extends|implements|permits|[<{(]|$))")
_LOOKALIKE = str.maketrans({"I": "l", "1": "l", "|": "l", "O": "o", "0": "o"})


def _identity(line: str) -> str | None:
    """Declared package / type name on a (whitespace-free) line, folded so that OCR
    look-alikes (l/I/1, O/0, letter case) do not count as a different name."""
    m = _IDENT_RE.match(line)
    if not m:
        return None
    return (m.group(1) or m.group(2)).translate(_LOOKALIKE).lower()


def _similar(a: str, b: str) -> bool:
    import difflib

    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.85


def _crumb_tail(crumb: list[str]) -> tuple[str, ...]:
    return tuple(crumb[-3:]) if crumb else ()


def segment(reads: list[FrameRead], cw: float, origin: float) -> list[Doc]:
    docs: list[Doc] = []
    last: Doc | None = None
    for fi, fr in enumerate(reads):
        if not fr.rows:
            continue
        lines = {}
        for r in fr.rows:
            indent, text = r.text(cw, origin)
            lines[r.n] = (indent, fix_line(text))
        rows = {n: norm(t) for n, (_, t) in lines.items()}
        hint = fr.name_hint
        tail = _crumb_tail(fr.crumb)

        target: Doc | None = None
        if hint:
            same = [d for d in docs if d.key == hint and (not tail or not d.crumb or d.crumb == tail)]
            scored = []
            clash = False
            for d in same:
                ov, ag = _agreement(d, rows)
                if ag < 0:
                    clash = True
                elif ov >= 3:
                    if ag / ov >= 0.6:
                        scored.append((ag / ov, d.frames[-1], d))
                    else:
                        clash = True
            if scored:
                target = max(scored, key=lambda s: (s[0], s[1]))[2]
            elif same and not clash:
                target = max(same, key=lambda d: d.frames[-1])
        else:
            if last is not None:
                ov, ag = _agreement(last, rows)
                if ag >= 0 and (ov >= 3 and ag / ov >= 0.6 or ov < 3):
                    target = last
            if target is None:
                best = None
                for d in docs:
                    ov, ag = _agreement(d, rows)
                    if ag >= 0 and ov >= 3 and ag / ov >= 0.7 and (best is None or ag / ov > best[0]):
                        best = (ag / ov, d)
                target = best[1] if best else None

        if target is None:
            target = Doc(id=len(docs), key=hint, crumb=tail)
            docs.append(target)
        if hint:
            target.hints[hint] += 1
        if fr.crumb:
            target.crumbs[tuple(fr.crumb)] += 1
        if not target.crumb and tail:
            target.crumb = tail
        target.frames.append(fi)
        for r in fr.rows:
            indent, text = lines[r.n]
            fit = max((abs(sg.fit) for sg in r.segs), default=0.0)
            target.obs.setdefault(r.n, []).append(
                Obs(fi, text, indent, r.score, fr.moving, fit))
        last = target
    return _merge(docs)


def _merge(docs: list[Doc]) -> list[Doc]:
    """Join documents that turned out to be the same file (revisited later, or seen without a hint)."""
    out: list[Doc] = []
    for d in sorted(docs, key=lambda d: -sum(len(v) for v in d.obs.values())):
        rows = {n: d.best(n) for n in d.obs}
        home = None
        for o in out:
            if d.key and o.key and d.key != o.key:
                continue
            ov, ag = _agreement(o, rows)
            # merging two different files is worse than leaving a duplicate: be strict
            if ag >= 0 and ov >= 3 and ov - ag <= ov * 0.05:
                home = o
                break
        if home is None:
            out.append(d)
            continue
        for n, obs in d.obs.items():
            home.obs.setdefault(n, []).extend(obs)
        home.frames.extend(d.frames)
        home.hints.update(d.hints)
        home.crumbs.update(d.crumbs)
    for i, d in enumerate(out):
        d.id = i
        d.frames.sort()
    return out


# ---------------------------------------------------------------- voting
def vote(doc: Doc, reads: list[FrameRead]) -> None:
    if not doc.obs:
        return
    last = max(doc.obs)
    doc.lines = []
    for n in range(1, last + 1):
        obs = doc.obs.get(n)
        if not obs:
            doc.lines.append(Line(n=n, flags=["missing"]))
            continue
        groups: dict[str, list[Obs]] = {}
        for o in obs:
            groups.setdefault(norm(o.text), []).append(o)

        def weight(os: list[Obs]) -> float:
            return sum(o.score * (0.6 if o.moving else 1.0) for o in os)

        total = sum(weight(g) for g in groups.values())
        win_key = max(groups, key=lambda k: weight(groups[k]))
        win = groups[win_key]
        text = Counter(o.text for o in win).most_common(1)[0][0]
        indent = Counter(o.indent for o in win).most_common(1)[0][0]
        best = max(win, key=lambda o: (not o.moving, o.score))
        ln = Line(n=n, text=text, indent=indent, conf=round(weight(win) / total, 3) if total else 0,
                  votes=len(obs), frame=best.frame)
        top = max(o.score for o in win)
        if ln.conf < 0.75:
            ln.flags.append("conflict")
        if top < 0.85:
            ln.flags.append("low-score")
        elif len(obs) == 1 and top < 0.97 and text:
            ln.flags.append("single")
        if CJK_RE.search(text) and top < 0.95:
            ln.flags.append("cjk")
        if min(o.fit for o in win) > 0.6:
            ln.flags.append("width")      # the text does not explain the ink: a glyph missing or extra
        if len({o.indent for o in win}) > 1 and Counter(o.indent for o in win).most_common(1)[0][1] < len(win) * 0.7:
            ln.flags.append("indent")
        doc.lines.append(ln)


# ---------------------------------------------------------------- naming
_TYPE_DECL = re.compile(
    r"^\s*(public\s+)?(?:(?:abstract|final|sealed|non-sealed|static|strictfp)\s+)*"
    r"(?:class|interface|enum|record|@interface)\s+([A-Za-z_$][\w$]*)")
_PACKAGE = re.compile(r"^\s*package\s*([a-z_][\w.]*)\s*[;；]")


def java_identity(text: str) -> tuple[str | None, str | None]:
    """(package, top-level type name) — the public type wins over helper types."""
    pkg = None
    found: list[tuple[bool, str]] = []
    for ln in text.split("\n"):
        if pkg is None:
            m = _PACKAGE.match(ln)
            if m:
                pkg = m.group(1)
        m = _TYPE_DECL.match(ln)
        if m and (len(ln) - len(ln.lstrip())) == 0:
            found.append((bool(m.group(1)), m.group(2)))
    pub = [n for p, n in found if p]
    return pkg, (pub[0] if pub else (found[0][1] if found else None))


_CONFUSE = str.maketrans({"I": "l", "1": "l", "|": "l", "O": "o", "0": "o"})


def reconcile_name(a: str, b: str) -> str | None:
    """Two readings of one identifier (code vs tab/breadcrumb) that differ only in l/I/1, O/0:
    rebuild it position by position. None if they are really different names."""
    if len(a) != len(b) or a.translate(_CONFUSE) != b.translate(_CONFUSE):
        return None
    out = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            out.append(x)
            continue
        prev = out[-1] if out else ""
        nxt = a[i + 1] if i + 1 < len(a) else ""
        letters = [c for c in (x, y) if c.isalpha()]
        if len(letters) == 1 and prev.isalpha():
            out.append(letters[0])                 # a digit inside a word is the misread
        elif {x, y} <= {"I", "l"}:
            # camel case: a capital starts a word; `l` + consonant does not start one
            out.append("I" if prev.islower() and nxt and nxt not in "aeiouy" else "l")
        else:
            out.append(letters[0] if letters else x)
    return "".join(out)


def resolve_path(doc: Doc) -> None:
    hint = doc.hints.most_common(1)[0][0] if doc.hints else None
    crumb = list(doc.crumbs.most_common(1)[0][0]) if doc.crumbs else []
    pkg, typ = java_identity(doc.text)
    if typ:
        for other in [hint[:-5] if hint and hint.endswith(".java") else None, crumb[-1] if crumb else None]:
            fixed = reconcile_name(typ, other) if other and other != typ else None
            if fixed and fixed != typ:
                doc.notes.append(f"name {typ} read as {other} elsewhere; using {fixed}")
                pat = re.compile(rf"(?<![\w$]){re.escape(typ)}(?![\w$])")
                for ln in doc.lines:
                    ln.text = pat.sub(fixed, ln.text)
                typ = fixed
                if hint and hint[:-5] != typ:
                    hint = f"{typ}.java"
                break
    is_java = typ is not None or (hint or "").endswith(".java")

    prefix: list[str] = []          # module directories from the navigation bar
    if crumb:
        segs = crumb[1:] if len(crumb) > 1 else crumb      # first segment is the project
        if "src" in segs:
            prefix = segs[:segs.index("src")]
        elif not is_java:
            fname = next((i for i in range(len(segs) - 1, -1, -1) if FILE_RE.fullmatch(segs[i])), None)
            if fname is not None:
                doc.path = "/".join(segs[:fname + 1])
                return

    if is_java:
        name = typ or (hint[:-5] if hint else None)
        if not name:
            doc.path, doc.path_sure = f"unknown/Doc{doc.id}.java", False
            doc.notes.append("no type declaration and no file name seen")
            return
        if hint and typ and hint != f"{typ}.java" and not reconcile_name(typ, hint[:-5]):
            doc.notes.append(f"tab said {hint}, code declares {typ}")
        pkg_dir = pkg.replace(".", "/") if pkg else ""
        if not pkg:
            doc.notes.append("no package line seen")
            doc.path_sure = False
        root = "src/main/java"
        if crumb and "src" in crumb:
            k = crumb.index("src")
            tail = crumb[k:k + 3]
            if len(tail) == 3 and tail[1] in ("main", "test") and tail[2] == "java":
                root = "/".join(tail)
        if not crumb:
            doc.path_sure = False
            doc.notes.append("module unknown (navigation bar not visible)")
        doc.path = "/".join(p for p in [*prefix, root, pkg_dir, f"{name}.java"] if p)
        return

    if crumb and "src" in crumb[1:]:
        segs = crumb[1:]
        fname = next((i for i in range(len(segs) - 1, -1, -1) if FILE_RE.fullmatch(segs[i])), None)
        if fname is not None:
            doc.path = "/".join(segs[:fname + 1])
            return
    doc.path = hint or f"unknown/doc{doc.id}.txt"
    doc.path_sure = False
    doc.notes.append("location unknown (navigation bar not visible)")


def _merge_same_path(docs: list[Doc], reads: list[FrameRead]) -> list[Doc]:
    """Two documents that resolved to the same file are one file seen twice, unless their
    shared lines disagree (then both are kept and flagged)."""
    by_path: dict[str, list[Doc]] = {}
    for d in docs:
        by_path.setdefault(d.path, []).append(d)
    out = []
    for group in by_path.values():
        group.sort(key=lambda d: -sum(len(v) for v in d.obs.values()))
        home = group[0]
        for d in group[1:]:
            rows = {n: d.best(n) for n in d.obs}
            ov, ag = _agreement(home, rows)
            if ov == 0 or ov - ag <= max(1, ov * 0.15):
                for n, obs in d.obs.items():
                    home.obs.setdefault(n, []).extend(obs)
                home.frames = sorted(home.frames + d.frames)
                home.hints.update(d.hints)
                home.crumbs.update(d.crumbs)
            else:
                out.append(d)
        if len(group) > 1:
            vote(home, reads)
        out.append(home)
    return out


def build(reads: list[FrameRead]) -> tuple[list[Doc], dict]:
    cw, origin = calibrate(reads)
    docs = segment(reads, cw, origin)
    for d in docs:
        vote(d, reads)
        resolve_path(d)
    docs = _merge_same_path(docs, reads)
    for d in docs:
        ext = "." + d.path.rsplit(".", 1)[-1] if "." in d.path else ""
        for ln, (text, changed) in zip(d.lines, fix_file([ln.text for ln in d.lines], ext)):
            if changed:
                ln.text = text
                ln.flags.append("autofix")
    docs = _merge_same_path(docs, reads)
    for d in docs:
        ext = "." + d.path.rsplit(".", 1)[-1] if "." in d.path else ""
        for ln, (text, changed) in zip(d.lines, fix_file([ln.text for ln in d.lines], ext)):
            if changed:
                ln.text = text
                ln.flags.append("autofix")
    # same path twice and not the same text: keep both, never overwrite silently
    seen: Counter = Counter()
    for d in sorted(docs, key=lambda d: -len(d.frames)):
        seen[d.path] += 1
        if seen[d.path] > 1:
            stem, dot, ext = d.path.rpartition(".")
            d.path = f"{stem}.dup{seen[d.path] - 1}.{ext}" if dot else f"{d.path}.dup{seen[d.path] - 1}"
            d.path_sure = False
            d.notes.append("another document resolved to the same path")
    docs = [d for d in docs if any(ln.text for ln in d.lines)]
    return docs, {"char_w": round(cw, 3), "origin": round(origin, 2)}
