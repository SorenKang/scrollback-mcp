"""Linux backend (X11): scrot for capture, xdotool for input, xclip for clipboard.

Wayland support is limited — xdotool only works under XWayland.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def _have(name: str) -> bool:
    return shutil.which(name) is not None


def virtual_screen() -> tuple[int, int, int, int]:
    if _have("xrandr"):
        out = _run(["xrandr"])
        for line in out.splitlines():
            if " connected" in line or " primary" in line:
                for tok in line.split():
                    if "x" in tok and tok[0].isdigit():
                        w, _, h = tok.partition("x")
                        try:
                            return 0, 0, int(w), int(h.rstrip("+*"))
                        except ValueError:
                            pass
    return 0, 0, 1920, 1080


def list_monitors() -> list[dict]:
    x, y, w, h = virtual_screen()
    return [{"index": 0, "x": x, "y": y, "width": w, "height": h,
             "work_x": x, "work_y": y, "work_width": w, "work_height": h, "primary": True}]


def capture(region=None, window=None, monitor=None, focus_first=False,
            all_screens=True) -> Image.Image:
    if region is None:
        region = virtual_screen()
    x, y, w, h = (int(v) for v in region)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        if _have("scrot"):
            _run(["scrot", "-o", "-a", f"{x},{y},{w},{h}", path])
        elif _have("import"):
            _run(["import", "-window", "root", "-crop", f"{w}x{h}+{x}+{y}", path])
        elif _have("gnome-screenshot"):
            _run(["gnome-screenshot", "-f", path])
        else:
            raise RuntimeError("no screenshot tool found: install scrot or imagemagick")
        return Image.open(path).convert("RGB")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def save(img: Image.Image, path: str, max_width: int = 0) -> str:
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    img.save(path)
    return path


def list_windows() -> list[dict]:
    if not _have("wmctrl"):
        return []
    out = _run(["wmctrl", "-lG"])
    wins = []
    for i, line in enumerate(out.splitlines()):
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        wins.append({"hwnd": parts[0], "x": int(parts[2]), "y": int(parts[3]),
                     "width": int(parts[4]), "height": int(parts[5]),
                     "title": parts[7], "class": "", "pid": 0,
                     "minimized": False, "foreground": i == 0})
    return wins


def find_window(title: str, exact: bool = False) -> dict | None:
    for w in list_windows():
        if title.lower() in w["title"].lower():
            return w
    return None


def focus_window(title: str, settle: float = 0.25) -> dict:
    w = find_window(title)
    if w and _have("wmctrl"):
        _run(["wmctrl", "-ia", w["hwnd"]])
        return {"ok": True, "focused": w["title"], "hwnd": w["hwnd"]}
    return {"ok": False, "reason": "window not found or wmctrl missing"}


def window_rect(title: str):
    w = find_window(title)
    return (w["x"], w["y"], w["width"], w["height"]) if w else None


def active_window() -> dict | None:
    out = _run(["xdotool", "getactivewindow", "getwindowname"]) if _have("xdotool") else ""
    return {"title": out, "hwnd": 0} if out else None


def cursor_pos() -> tuple[int, int]:
    if not _have("xdotool"):
        return 0, 0
    out = _run(["xdotool", "getmouselocation"])
    d = dict(p.split(":") for p in out.split() if ":" in p)
    return int(d.get("x", 0)), int(d.get("y", 0))


def move_to(x: int, y: int, duration: float = 0.0) -> tuple[int, int]:
    _run(["xdotool", "mousemove", str(x), str(y)])
    return x, y


def click(x=None, y=None, button: str = "left", clicks: int = 1, interval: float = 0.04) -> dict:
    if x is not None and y is not None:
        move_to(x, y)
    btn = {"left": "1", "middle": "2", "right": "3"}[button]
    _run(["xdotool", "click", "--repeat", str(clicks), btn])
    return {"ok": True, "pos": cursor_pos(), "button": button, "clicks": clicks}


def drag(x1: int, y1: int, x2: int, y2: int, duration: float = 0.5, button: str = "left") -> dict:
    move_to(x1, y1)
    _run(["xdotool", "mousedown", "1"])
    move_to(x2, y2)
    _run(["xdotool", "mouseup", "1"])
    return {"ok": True, "from": [x1, y1], "to": [x2, y2]}


def scroll(amount: int, x=None, y=None) -> dict:
    btn = "4" if amount > 0 else "5"
    for _ in range(abs(amount)):
        _run(["xdotool", "click", btn])
    return {"ok": True, "amount": amount}


def type_unicode(text: str, interval: float = 0.002) -> dict:
    _run(["xdotool", "type", "--clearmodifiers", "--delay", str(int(interval * 1000)), text])
    return {"ok": True, "chars": len(text), "method": "xdotool"}


def press(combo: str, times: int = 1, interval: float = 0.03) -> dict:
    keys = "+".join(combo.split("+"))
    for _ in range(times):
        _run(["xdotool", "key", keys])
    return {"ok": True, "combo": combo, "times": times}


def clipboard_get() -> str:
    return _run(["xclip", "-selection", "clipboard", "-o"]) if _have("xclip") else ""


def clipboard_set(text: str) -> bool:
    if not _have("xclip"):
        return False
    p = subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True)
    return p.returncode == 0


def paste_text(text: str, settle: float = 0.12) -> dict:
    prev = clipboard_get()
    if not clipboard_set(text):
        return type_unicode(text)
    press("ctrl+v")
    if prev:
        clipboard_set(prev)
    return {"ok": True, "chars": len(text), "method": "clipboard-paste"}


def type_text(text: str, method: str = "auto", interval: float = 0.002) -> dict:
    if method == "auto":
        method = "paste" if (len(text) > 24 or any(ord(c) > 127 for c in text) or "\n" in text) else "unicode"
    return paste_text(text) if method == "paste" else type_unicode(text, interval=interval)
