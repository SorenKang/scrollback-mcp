"""Score a rebuilt tree against the original source.

    python tools/evaluate.py <truth_dir> <rebuilt_dir> [--files list.txt] [--show 20]

Per file: exact-line accuracy (indentation included, trailing spaces ignored) and
character similarity. Files are matched by relative path; a rebuilt file at another
path but with the same name is reported as "misplaced".
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path


def lines(p: Path) -> list[str]:
    t = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").expandtabs(4)
    out = [ln.rstrip() for ln in t.split("\n")]
    while out and not out[-1]:
        out.pop()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("rebuilt")
    ap.add_argument("--files", help="manifest of relative paths that were recorded")
    ap.add_argument("--show", type=int, default=15, help="print up to N wrong lines")
    args = ap.parse_args()
    truth, reb = Path(args.truth), Path(args.rebuilt)
    if args.files:
        rels = [x.strip() for x in Path(args.files).read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        rels = sorted(p.relative_to(truth).as_posix() for p in truth.rglob("*") if p.is_file())
    by_name: dict[str, list[Path]] = {}
    for p in reb.rglob("*"):
        if p.is_file() and ".xocr" not in p.parts:
            by_name.setdefault(p.name, []).append(p)

    tot_lines = tot_ok = 0
    sims = []
    wrong: list[str] = []
    missing, misplaced = [], []
    for rel in rels:
        t = lines(truth / rel)
        r_path = reb / rel
        if not r_path.exists():
            alt = by_name.get(Path(rel).name)
            if alt:
                misplaced.append(f"{rel} -> {alt[0].relative_to(reb).as_posix()}")
                r_path = alt[0]
            else:
                missing.append(rel)
                tot_lines += len(t)
                sims.append(0.0)
                continue
        r = lines(r_path)
        ok = sum(1 for i, ln in enumerate(t) if i < len(r) and r[i] == ln)
        tot_lines += len(t)
        tot_ok += ok
        sims.append(difflib.SequenceMatcher(None, "\n".join(t), "\n".join(r), autojunk=False).ratio())
        for i, ln in enumerate(t):
            got = r[i] if i < len(r) else "<none>"
            if got != ln:
                wrong.append(f"{rel}:{i + 1}\n    want {ln!r}\n    got  {got!r}")
        if len(r) > len(t):
            wrong.append(f"{rel}: {len(r) - len(t)} extra lines")

    print(f"files: {len(rels)}  missing: {len(missing)}  misplaced: {len(misplaced)}")
    print(f"lines exact: {tot_ok}/{tot_lines} = {tot_ok / max(1, tot_lines):.4f}")
    print(f"char similarity (mean per file): {sum(sims) / max(1, len(sims)):.4f}")
    for m in missing:
        print("  MISSING", m)
    for m in misplaced:
        print("  MISPLACED", m)
    for w in wrong[: args.show]:
        print(" ", w)
    if len(wrong) > args.show:
        print(f"  … {len(wrong) - args.show} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
