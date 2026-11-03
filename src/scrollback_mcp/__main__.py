"""Command line.

    scrollback-mcp                              run the MCP server (stdio) — what MCP clients start
    scrollback-mcp rebuild VIDEO -o OUT         recording → project directory
    scrollback-mcp report OUT                   what still needs review
    scrollback-mcp crop OUT FILE START [END]    frame crop showing those lines
    scrollback-mcp set OUT FILE LINE TEXT       fix line(s); TEXT may hold \\n; '-' reads stdin
    scrollback-mcp verify OUT [--build]         syntax check (+ mvn compile)
    scrollback-mcp frame VIDEO SECONDS -o PNG   save one frame
    scrollback-mcp check [--fix-gpu]            environment diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _print(obj) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def _progress(p: dict) -> None:
    if p.get("stage") == "ocr":
        sys.stderr.write(f"\r  ocr {p['t']:>7.1f}/{p['seconds']}s  frames {p['frames']}  "
                         f"elapsed {p['elapsed']}s   ")
    else:
        sys.stderr.write(f"\n  {p['stage']}…")
    sys.stderr.flush()


def _fix_gpu() -> int:
    """Swap the CPU onnxruntime for onnxruntime-directml.

    rapidocr pulls in the CPU build; both provide the same `onnxruntime` package and pip
    never replaces one with the other, so the CPU build wins the import. Must run before
    onnxruntime is imported in this process: Windows locks loaded DLLs.
    """
    import subprocess

    py = sys.executable
    subprocess.run([py, "-m", "pip", "uninstall", "-y", "onnxruntime"], check=False)
    # the uninstall also removes files the DirectML build shares with it: reinstall in full
    r = subprocess.run([py, "-m", "pip", "install", "--force-reinstall", "--no-deps",
                        "onnxruntime-directml"], check=False)
    print("now run `scrollback-mcp check`: ocr_backend should be dml", file=sys.stderr)
    return r.returncode


def _check(fix_gpu: bool) -> int:
    if fix_gpu:
        return _fix_gpu()
    info: dict = {"version": __version__}
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["providers"] = ort.get_available_providers()
    except Exception as exc:  # noqa: BLE001
        info["onnxruntime_error"] = f"{type(exc).__name__}: {exc}"
    from . import ocr, verify

    info["ocr_backend"] = ocr.pick_backend()
    info["javac"] = verify.find_javac()
    info["mvn"] = verify.find_mvn()
    tips = []
    if info["ocr_backend"] == "cpu" and sys.platform == "win32":
        try:
            from importlib.metadata import version

            version("onnxruntime-directml")
            tips.append("onnxruntime-directml is installed but the CPU onnxruntime shadows it: "
                        "run `scrollback-mcp check --fix-gpu`")
        except Exception:  # noqa: BLE001
            tips.append('CPU only. For GPU on Windows: pip install "scrollback-mcp[gpu]" '
                        "then `scrollback-mcp check --fix-gpu`")
    if not info["javac"]:
        tips.append("javac not found: Java syntax checking is disabled (set JAVA_HOME)")
    if tips:
        info["tips"] = tips
    _print(info)
    return 0


def main() -> int:
    if len(sys.argv) == 1 or sys.argv[1] in ("serve", "--stdio"):
        from .server import mcp

        mcp.run(transport="stdio")
        return 0

    ap = argparse.ArgumentParser(prog="scrollback-mcp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("rebuild", help="recording → project directory")
    p.add_argument("video")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--fps", type=float, default=10.0, help="frames sampled per second (default 10)")
    p.add_argument("--no-verify", action="store_true")

    p = sub.add_parser("report", help="print the review report")
    p.add_argument("out")
    p.add_argument("--json", action="store_true", help="the review queue as JSON")

    p = sub.add_parser("crop", help="frame crop showing lines START..END of FILE")
    p.add_argument("out")
    p.add_argument("file")
    p.add_argument("start", type=int)
    p.add_argument("end", type=int, nargs="?")

    p = sub.add_parser("set", help="replace line(s) of FILE starting at LINE")
    p.add_argument("out")
    p.add_argument("file")
    p.add_argument("line", type=int)
    p.add_argument("text", help="new text; \\n separates lines; '-' reads stdin")

    p = sub.add_parser("verify", help="syntax-check the rebuilt tree")
    p.add_argument("out")
    p.add_argument("--build", action="store_true", help="also run `mvn compile`")

    p = sub.add_parser("frame", help="save the frame at SECONDS")
    p.add_argument("video")
    p.add_argument("seconds", type=float)
    p.add_argument("-o", "--out", default="frame.png")

    p = sub.add_parser("check", help="environment diagnostics")
    p.add_argument("--fix-gpu", action="store_true",
                   help="remove the CPU onnxruntime that shadows onnxruntime-directml")

    a = ap.parse_args()
    if a.cmd == "rebuild":
        from . import rebuild

        res = rebuild.run(a.video, a.out, sample_fps=a.fps, progress=_progress,
                          verify=not a.no_verify)
        sys.stderr.write("\n")
        res.get("verify", {}).pop("by_file", None)
        _print(res)
        return 0
    if a.cmd == "report":
        from . import review

        if a.json:
            _print(review.queue(a.out))
        else:
            sys.stdout.write((Path(a.out) / ".xocr" / "REPORT.md").read_text(encoding="utf-8"))
        return 0
    if a.cmd == "crop":
        from . import review

        _print(review.crop(a.out, a.file, a.start, a.end))
        return 0
    if a.cmd == "set":
        from . import review

        text = sys.stdin.read().rstrip("\n") if a.text == "-" else a.text.replace("\\n", "\n")
        _print(review.set_lines(a.out, a.file, a.line, text))
        return 0
    if a.cmd == "verify":
        from . import review, verify

        res = verify.check(a.out, build=a.build)
        review.write_report(a.out)
        _print(res)
        return 0
    if a.cmd == "frame":
        import cv2

        from . import video

        fps = video.info(a.video)["fps"]
        cv2.imencode(".png", video.frame_at(a.video, int(a.seconds * fps)))[1].tofile(a.out)
        _print({"ok": True, "path": str(Path(a.out).resolve())})
        return 0
    if a.cmd == "check":
        return _check(a.fix_gpu)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
