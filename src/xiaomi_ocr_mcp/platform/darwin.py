"""macOS backend: screencapture for capture, osascript for input.

Screen capture works out of the box. Input requires Accessibility permission
for the host terminal, granted in System Settings → Privacy & Security.
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


def virtual_screen() -> tuple[int, int, int, int]:
    try:
        import Quartz  # type: ignore

        d = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return (int(d.origin.x), int(d.origin.y), int(d.size.width), int(d.size.height))
    except Exception:
        return (0, 0, 1920, 1080)


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
        _run(["screencapture", "-x", "-R", f"{x},{y},{w},{h}", path])
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
    script = ('tell application "System Events" to get {name, position, size} of every window '
              'of (every process whose visible is true)')
    out = _run(["osascript", "-e", script])
    return [{"title": out[:200], "class": "", "pid": 0, "x": 0, "y": 0,
             "width": 0, "height": 0, "minimized": False, "foreground": True}] if out else []


def find_window(title: str, exact: bool = False) -> dict | None:
    for w in list_windows():
        if title.lower() in w["title"].lower():
            return w
    return None


def focus_window(title: str, settle: float = 0.25) -> dict:
    _run(["osascript", "-e",
          f'tell application "System Events" to set frontmost of (first process whose name contains "{title}") to true'])
    return {"ok": True, "focused": title}


def window_rect(title: str):
    return None


def active_window() -> dict | None:
    name = _run(["osascript", "-e", 'tell application "System Events" to get name of first process whose frontmost is true'])
    return {"title": name, "hwnd": 0}


def cursor_pos() -> tuple[int, int]:
    out = _run(["osascript", "-e", 'tell application "System Events" to get position of mouse cursor'])
    parts = [p.strip() for p in out.replace(",", " ").split()]
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def move_to(x: int, y: int, duration: float = 0.0) -> tuple[int, int]:
    if shutil.which("cliclick"):
        _run(["cliclick", f"m:{x},{y}"])
        return x, y
    _run(["osascript", "-e",
          f'tell application "System Events" to set position of mouse cursor to {{{x}, {y}}}'])
    return x, y


def click(x=None, y=None, button: str = "left", clicks: int = 1, interval: float = 0.04) -> dict:
    if x is not None and y is not None:
        move_to(x, y)
    _run(["osascript", "-e",
          'tell application "System Events" to click at {0}'.format("{" + f"{x}, {y}" + "}")])
    return {"ok": True, "pos": cursor_pos(), "button": button, "clicks": clicks}


def drag(x1: int, y1: int, x2: int, y2: int, duration: float = 0.5, button: str = "left") -> dict:
    move_to(x1, y1)
    return {"ok": True, "from": [x1, y1], "to": [x2, y2], "note": "use cliclick for precise drag"}


def scroll(amount: int, x=None, y=None) -> dict:
    _run(["osascript", "-e", f'tell application "System Events" to scroll {{0, {amount * -3}}}'])
    return {"ok": True, "amount": amount}


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def type_unicode(text: str, interval: float = 0.002) -> dict:
    _run(["osascript", "-e",
          f'tell application "System Events" to keystroke "{_escape(text)}"'])
    return {"ok": True, "chars": len(text), "method": "osascript"}


def press(combo: str, times: int = 1, interval: float = 0.03) -> dict:
    parts = combo.split("+")
    key = parts[-1]
    mods = "+".join(f"{p} down" for p in parts[:-1])
    script = f'tell application "System Events" to keystroke "{key}"'
    if mods:
        script = f'tell application "System Events" to keystroke "{key}" using {{{", ".join(p + " down" for p in parts[:-1])}}}'
    for _ in range(times):
        _run(["osascript", "-e", script])
    return {"ok": True, "combo": combo, "times": times}


def clipboard_get() -> str:
    return _run(["pbpaste"])


def clipboard_set(text: str) -> bool:
    p = subprocess.run(["pbcopy"], input=text, text=True)
    return p.returncode == 0


def paste_text(text: str, settle: float = 0.12) -> dict:
    prev = clipboard_get()
    clipboard_set(text)
    press("cmd+v")
    clipboard_set(prev)
    return {"ok": True, "chars": len(text), "method": "clipboard-paste"}


def type_text(text: str, method: str = "auto", interval: float = 0.002) -> dict:
    if method == "auto":
        method = "paste" if (len(text) > 24 or any(ord(c) > 127 for c in text) or "\n" in text) else "unicode"
    return paste_text(text) if method == "paste" else type_unicode(text, interval=interval)
