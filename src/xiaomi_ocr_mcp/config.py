"""Runtime configuration: model location, backend selection, feature flags.

Everything is overridable by environment variable so the same wheel works for
everyone without editing files.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "xiaomi-ocr-mcp"


def home() -> Path:
    """Cache root. `XIAOMI_OCR_HOME` wins, else the platform cache dir."""
    env = os.environ.get("XIAOMI_OCR_HOME")
    if env:
        p = Path(env).expanduser()
    else:
        try:
            from platformdirs import user_cache_dir

            p = Path(user_cache_dir(APP_NAME))
        except Exception:  # noqa: BLE001
            p = Path(os.path.expanduser("~")) / f".{APP_NAME}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def vlm_model_dir() -> Path:
    env = os.environ.get("XIAOMI_OCR_MODEL_DIR")
    return Path(env).expanduser() if env else home() / "models" / "Xiaomi-OCR-0"


def vlm_enabled() -> bool:
    """VLM 通道默认关闭：1.7GB 权重 + torch，绝大多数场景快通道就够。

    显式开：XIAOMI_OCR_VLM=1；或者模型目录已存在也会自动启用。
    """
    flag = os.environ.get("XIAOMI_OCR_VLM", "").lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    return (vlm_model_dir() / "config.json").exists()


def fast_model() -> str:
    return os.environ.get("RAPID_CODE_MODEL", "small").lower()


def gpu_mode() -> str:
    return os.environ.get("RAPID_CODE_GPU", "auto").lower()


def shots_dir() -> Path:
    env = os.environ.get("XIAOMI_OCR_SHOTS")
    p = Path(env).expanduser() if env else home() / "shots"
    p.mkdir(parents=True, exist_ok=True)
    return p
