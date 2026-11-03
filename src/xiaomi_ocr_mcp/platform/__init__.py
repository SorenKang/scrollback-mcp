"""Cross-platform desktop primitives: capture, windows, mouse, keyboard, clipboard.

Windows has a native ctypes implementation (no extra deps).
macOS / Linux shell out to system tools so the package stays dependency-free.
"""

from __future__ import annotations

import sys

PLATFORM = "windows" if sys.platform == "win32" else ("darwin" if sys.platform == "darwin" else "linux")

if PLATFORM == "windows":
    from .windows_screen import (  # noqa: F401
        active_window,
        capture,
        find_window,
        focus_window,
        list_monitors,
        list_windows,
        save,
        virtual_screen,
        window_rect,
    )
    from .windows_input import (  # noqa: F401
        click,
        clipboard_get,
        clipboard_set,
        cursor_pos,
        drag,
        move_to,
        paste_text,
        press,
        scroll,
        type_text,
        type_unicode,
    )
elif PLATFORM == "darwin":
    from .darwin import *  # noqa: F401,F403
else:
    from .linux import *  # noqa: F401,F403

__all__ = [
    "PLATFORM", "capture", "save", "list_monitors", "virtual_screen",
    "list_windows", "find_window", "focus_window", "window_rect", "active_window",
    "cursor_pos", "move_to", "click", "drag", "scroll",
    "type_text", "type_unicode", "paste_text", "press",
    "clipboard_get", "clipboard_set",
]
