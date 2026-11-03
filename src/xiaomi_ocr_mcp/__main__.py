"""CLI entry point.

    xiaomi-ocr-mcp                 # start the stdio MCP server (what clients invoke)
    xiaomi-ocr-mcp --check         # print environment / backend diagnostics
    xiaomi-ocr-mcp --download-model # fetch Xiaomi-OCR-0 (~1.7 GB) for the VLM channel
    xiaomi-ocr-mcp --selftest      # capture + OCR smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__, config


def _check() -> int:
    info: dict = {"version": __version__}
    try:
        from .platform import PLATFORM

        info["platform"] = PLATFORM
    except Exception as exc:  # noqa: BLE001
        info["platform_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["providers"] = ort.get_available_providers()
    except Exception as exc:  # noqa: BLE001
        info["onnxruntime_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from . import engine_fast

        info["fast_backend"] = engine_fast._pick_backend()
    except Exception as exc:  # noqa: BLE001
        info["fast_backend_error"] = f"{type(exc).__name__}: {exc}"

    info["vlm_enabled"] = config.vlm_enabled()
    info["vlm_model_dir"] = str(config.vlm_model_dir())
    info["cache_dir"] = str(config.home())

    tips = []
    if info.get("fast_backend") == "cpu":
        tips.append(
            "Running on CPU (~2 s/screen). For ~0.23 s/screen install the GPU extra: "
            'pip install "xiaomi-ocr-mcp[gpu]" (Windows/DirectML)'
        )
    if not info["vlm_enabled"]:
        tips.append(
            "VLM channel is off (fast path only). Enable with: "
            "pip install \"xiaomi-ocr-mcp[vlm]\" && xiaomi-ocr-mcp --download-model"
        )
    if tips:
        info["tips"] = tips

    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def _download_model() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is missing. Install with:\n"
              "  pip install 'xiaomi-ocr-mcp[vlm]'", file=sys.stderr)
        return 1
    dest = config.vlm_model_dir()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading Xiaomi-OCR-0 → {dest}", file=sys.stderr)
    snapshot_download(
        repo_id="xiaomi-research/Xiaomi-OCR-0",
        local_dir=str(dest),
        allow_patterns=["config.json", "model.safetensors", "processor_config.json",
                        "tokenizer*.json", "chat_template.jinja", "*.json"],
        max_workers=8,
    )
    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"done: {total / 1024 ** 3:.2f} GB → {dest}", file=sys.stderr)
    return 0


def _selftest() -> int:
    from . import engine_fast
    from .platform import capture, save

    shot = config.shots_dir() / "selftest.png"
    img = capture()
    save(img, str(shot))
    t0 = time.time()
    r = engine_fast.ocr_code(img)
    print(f"capture {img.width}x{img.height} → {shot}", file=sys.stderr)
    print(f"OCR {time.time() - t0:.2f}s, {r['lines']} lines, score {r['avg_score']:.3f}", file=sys.stderr)
    print(r["code"][:800], file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="xiaomi-ocr-mcp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--check", action="store_true", help="print environment diagnostics")
    ap.add_argument("--download-model", action="store_true", help="download Xiaomi-OCR-0 weights")
    ap.add_argument("--selftest", action="store_true", help="capture + OCR smoke test")
    args = ap.parse_args()

    if args.check:
        return _check()
    if args.download_model:
        return _download_model()
    if args.selftest:
        return _selftest()

    from .server import mcp

    print(f"[xiaomi-ocr-mcp] starting stdio server (cache={config.home()}, "
          f"vlm={'on' if config.vlm_enabled() else 'off'})", file=sys.stderr)
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
