"""MCP server: screen OCR + desktop control.

Transport: stdio (JSON-RPC). Logs go to stderr only — stdout must stay clean.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageGrab

from . import config
from .platform import (  # noqa: F401
    PLATFORM,
    active_window,
    capture,
    click,
    clipboard_get as _clip_get,
    clipboard_set as _clip_set,
    cursor_pos,
    drag,
    find_window,
    focus_window as _focus_window,
    list_monitors,
    list_windows as _list_windows,
    move_to,
    paste_text,
    press,
    save,
    scroll,
    type_text as _type_text,
    virtual_screen,
)

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="[xiaomi-ocr-mcp] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:  # mcp >= 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore

mcp = _Server("xiaomi-ocr-desktop")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]


def _shot_path(prefix: str = "shot") -> Path:
    return config.shots_dir() / f"{prefix}-{_ts()}.png"


def _trim(text: str, limit: int = 20000) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated, {len(text)} chars]"


# ---------------------------------------------------------------- capture
@mcp.tool()
def screen_info() -> dict[str, Any]:
    """Monitors, virtual desktop size, active window and cursor position. Call this before clicking."""
    return {
        "platform": PLATFORM,
        "monitors": list_monitors(),
        "virtual_screen": dict(zip(("x", "y", "width", "height"), virtual_screen())),
        "active_window": active_window(),
        "cursor": dict(zip(("x", "y"), cursor_pos())),
    }


@mcp.tool()
def screenshot(monitor: int | None = None, window: str | None = None,
               region: list[int] | None = None, focus_first: bool = False,
               path: str | None = None) -> dict[str, Any]:
    """Capture the screen to a PNG file and return its absolute path. Fastest way to look at the screen.

    region=[x, y, w, h] in physical pixels; window=title substring; monitor=index; all empty = whole desktop.
    """
    img = capture(region=tuple(region) if region else None, window=window,
                  monitor=monitor, focus_first=focus_first)
    out = Path(path) if path else _shot_path("shot")
    out.parent.mkdir(parents=True, exist_ok=True)
    save(img, str(out))
    return {"ok": True, "path": str(out), "width": img.width, "height": img.height}


@mcp.tool()
def read_code(monitor: int | None = None, window: str | None = None,
              region: list[int] | None = None, focus_first: bool = False,
              engine: str = "auto", lang: str = "ch", blank_lines: bool = True) -> dict[str, Any]:
    """Read code off the screen. Fast path (PP-OCR + bbox rebuild) takes ~0.2-0.5 s and keeps indentation.

    engine: auto (fast, falls back to VLM when unsure) | fast | vlm
    """
    from . import engine_fast

    img = capture(region=tuple(region) if region else None, window=window,
                  monitor=monitor, focus_first=focus_first)
    shot = _shot_path("code")
    save(img, str(shot))
    out: dict[str, Any] = {"ok": True, "image": str(shot), "size": [img.width, img.height]}

    if engine != "vlm":
        fast = engine_fast.ocr_code(img, lang=lang, blank_lines=blank_lines)
        out.update({
            "channel": "fast", "code": _trim(fast["code"]), "lines": fast["lines"],
            "chars": len(fast["code"]), "boxes": fast["items"],
            "avg_score": fast["avg_score"], "min_score": fast["min_score"],
            "cjk_ratio": fast["cjk_ratio"], "seconds": fast["seconds"],
        })
        good = (fast["items"] >= 2 and fast["avg_score"] >= 0.86
                and len(fast["code"]) >= 12 and fast["cjk_ratio"] <= 0.03
                and not fast["quote_unbalanced"])
        if engine == "fast" or good:
            return out
        out["fallback_reason"] = "fast path low confidence, retried with VLM"

    if not config.vlm_enabled():
        out["ok"] = False
        out["hint"] = ("VLM channel disabled. Set XIAOMI_OCR_VLM=1 and download the model "
                       "(`xiaomi-ocr-mcp --download-model`), or use engine=\"fast\".")
        return out

    from . import engine_vlm

    res = engine_vlm.get_engine().run(img, mode="code", max_new_tokens=2048)
    code = engine_vlm.clean_output(res["text"], "code")
    out.update({"channel": "vlm", "code": _trim(code),
                "lines": code.count("\n") + 1 if code else 0, "chars": len(code),
                "avg_score": None, "min_score": None, "seconds": res["seconds"]})
    return out


@mcp.tool()
def see_screen(monitor: int | None = None, window: str | None = None,
               region: list[int] | None = None, mode: str = "text",
               focus_first: bool = False, max_new_tokens: int = 3072) -> dict[str, Any]:
    """Capture + Xiaomi-OCR-0: reads text, tables, formulas, multi-column documents.

    mode: text | code | document | table | formula
    For plain code prefer read_code (20x faster).
    """
    img = capture(region=tuple(region) if region else None, window=window,
                  monitor=monitor, focus_first=focus_first)
    shot = _shot_path("see")
    save(img, str(shot))
    if not config.vlm_enabled():
        return {"ok": False, "image": str(shot),
                "hint": "VLM channel disabled; set XIAOMI_OCR_VLM=1 or use read_code."}
    from . import engine_vlm

    res = engine_vlm.get_engine().run(img, mode=mode, max_new_tokens=max_new_tokens)
    text = res["text"]
    if mode == "code":
        text = engine_vlm.clean_output(text, mode)
    return {"ok": True, "text": _trim(text), "chars": len(text), "mode": mode,
            "seconds": res["seconds"], "device": res["device"], "image": str(shot)}


@mcp.tool()
def ocr_image(image_path: str, mode: str = "text", prompt: str | None = None,
              max_new_tokens: int = 3072) -> dict[str, Any]:
    """OCR a local image file. mode as in see_screen; prompt overrides the built-in instruction."""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not config.vlm_enabled():
        from . import engine_fast

        with Image.open(p) as im:
            r = engine_fast.ocr_code(im.convert("RGB"))
        return {"ok": True, "text": _trim(r["code"]), "chars": len(r["code"]),
                "mode": "fast", "seconds": r["seconds"]}
    from . import engine_vlm

    with Image.open(p) as im:
        res = engine_vlm.get_engine().run(im.convert("RGB"), mode=mode,
                                          prompt=prompt, max_new_tokens=max_new_tokens)
    text = res["text"]
    if mode == "code":
        text = engine_vlm.clean_output(text, mode)
    return {"ok": True, "text": _trim(text), "chars": len(text), "mode": res["mode"],
            "seconds": res["seconds"], "device": res["device"]}


@mcp.tool()
def ocr_clipboard(mode: str = "text") -> dict[str, Any]:
    """OCR the image currently on the clipboard (after a screenshot shortcut)."""
    data = ImageGrab.grabclipboard()
    if data is None:
        return {"ok": False, "reason": "no image on clipboard", "text": ""}
    if isinstance(data, list):
        imgs = [Image.open(x) for x in data if str(x).lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        if not imgs:
            return {"ok": False, "reason": "clipboard holds files, none of them images", "text": ""}
        img = imgs[0]
    else:
        img = data
    if not config.vlm_enabled():
        from . import engine_fast

        r = engine_fast.ocr_code(img.convert("RGB"))
        return {"ok": True, "text": _trim(r["code"]), "chars": len(r["code"]), "seconds": r["seconds"]}
    from . import engine_vlm

    res = engine_vlm.get_engine().run(img.convert("RGB"), mode=mode)
    text = res["text"]
    if mode == "code":
        text = engine_vlm.clean_output(text, mode)
    return {"ok": True, "text": _trim(text), "chars": len(text),
            "seconds": res["seconds"], "device": res["device"]}


# ---------------------------------------------------------------- windows
@mcp.tool()
def list_windows() -> list[dict[str, Any]]:
    """List visible top-level windows: title, class, pid, rect, foreground/minimized flags."""
    return _list_windows()


@mcp.tool()
def focus_window(title: str) -> dict[str, Any]:
    """Bring the window whose title contains `title` to the foreground. Do this before typing."""
    return _focus_window(title)


# ---------------------------------------------------------------- input
@mcp.tool()
def mouse_move(x: int, y: int, duration: float = 0.0) -> dict[str, Any]:
    """Move the cursor to physical pixel (x, y). duration>0 animates."""
    return {"ok": True, "pos": move_to(x, y, duration=duration)}


@mcp.tool()
def mouse_click(x: int | None = None, y: int | None = None, button: str = "left",
                clicks: int = 1) -> dict[str, Any]:
    """Click at (x, y) or at the current position. button=left|right|middle; clicks=2 double-clicks."""
    return click(x, y, button=button, clicks=clicks)


@mcp.tool()
def mouse_drag(x1: int, y1: int, x2: int, y2: int, duration: float = 0.5,
               button: str = "left") -> dict[str, Any]:
    """Press, move, release — selecting text, dragging files, pulling scrollbars."""
    return drag(x1, y1, x2, y2, duration=duration, button=button)


@mcp.tool()
def mouse_scroll(amount: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
    """Scroll the wheel. amount>0 up, <0 down, 1 = one notch."""
    return scroll(amount, x, y)


@mcp.tool()
def type_text(text: str, method: str = "auto") -> dict[str, Any]:
    """Type into the focused window. method: auto|paste|unicode.

    auto uses clipboard+paste for long/non-ASCII/multi-line text, unicode injection otherwise.
    """
    return _type_text(text, method=method)


@mcp.tool()
def press_keys(combo: str, times: int = 1) -> dict[str, Any]:
    """Press a key combo: "ctrl+s", "alt+tab", "enter", "escape", "ctrl+shift+p", "cmd+a"."""
    return press(combo, times=times)


@mcp.tool()
def type_code(code: str, method: str = "paste") -> dict[str, Any]:
    """Paste a code block into the focused window, indentation and newlines preserved."""
    return _type_text(code, method=method)


@mcp.tool()
def clipboard_set(text: str) -> dict[str, Any]:
    """Put text on the clipboard. Fallback when synthetic input is blocked — user pastes manually."""
    return {"ok": _clip_set(text), "chars": len(text)}


@mcp.tool()
def clipboard_get() -> dict[str, Any]:
    """Read the clipboard as text."""
    t = _clip_get()
    return {"ok": True, "text": _trim(t), "chars": len(t)}


@mcp.tool()
def wait(seconds: float = 1.0) -> dict[str, Any]:
    """Sleep, for UI loads / compiles / animations."""
    time.sleep(max(0.0, min(seconds, 120.0)))
    return {"ok": True, "waited": seconds}


@mcp.tool()
def wait_for_window(title: str, timeout: float = 15.0, interval: float = 0.4) -> dict[str, Any]:
    """Poll until a window whose title contains `title` appears."""
    deadline = time.time() + max(0.5, timeout)
    while time.time() < deadline:
        w = find_window(title)
        if w:
            return {"ok": True, "window": w}
        time.sleep(interval)
    return {"ok": False, "reason": f"no window matching {title!r} within {timeout}s"}


@mcp.tool()
def input_selftest(timeout: float = 12.0, text: str = "SELFTEST-中文-42") -> dict[str, Any]:
    """Verify the synthetic-input path end to end: opens a Tk window, clicks a button and types, reads back.

    If mouse_ok and key_ok are both true but real targets ignore input, the OS is
    filtering synthetic events across process boundaries (sandbox / remote session).
    """
    import threading

    result: dict[str, Any] = {}
    done = threading.Event()

    def worker() -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.title("OCR-MCP-INPUT-SELFTEST")
            root.geometry("420x170+300+260")
            hits = {"click": 0}
            btn = tk.Button(root, text="CLICK",
                            command=lambda: hits.__setitem__("click", hits["click"] + 1))
            btn.pack(pady=8, ipadx=30, ipady=6)
            entry = tk.Entry(root, width=42)
            entry.pack(pady=8)
            entry.focus_force()
            for _ in range(40):
                root.update()
                time.sleep(0.05)
            _focus_window("OCR-MCP-INPUT-SELFTEST")
            root.update()
            time.sleep(0.35)
            x = btn.winfo_rootx() + btn.winfo_width() // 2
            y = btn.winfo_rooty() + btn.winfo_height() // 2
            click(x, y)
            for _ in range(20):
                root.update()
                time.sleep(0.02)
            entry.focus_force()
            root.update()
            time.sleep(0.15)
            from .platform import type_unicode

            type_unicode(text)
            for _ in range(40):
                root.update()
                time.sleep(0.02)
            result.update({"mouse_ok": hits["click"] > 0,
                           "key_ok": entry.get() == text, "typed": entry.get()})
            root.destroy()
        except Exception as exc:  # noqa: BLE001
            result.update({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    done.wait(timeout=max(3.0, timeout))
    ok = bool(result.get("mouse_ok")) and bool(result.get("key_ok"))
    return {"ok": ok, **result}
