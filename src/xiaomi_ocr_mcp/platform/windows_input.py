"""Windows 键鼠注入 / 剪贴板（纯 ctypes，无第三方依赖）。

设计要点：
- 进程启动即声明 Per-Monitor-V2 DPI 感知，保证坐标与物理像素一致（否则 150% 缩放下点击会偏移）。
- 鼠标：SetCursorPos + SendInput 按键事件。键盘：SendInput，支持 VK 组合键与 Unicode 直投。
- 剪贴板：ctypes 直连 user32，不依赖 pyperclip / tkinter。
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# ---------------------------------------------------------------- DPI 感知
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:  # noqa: BLE001
    try:
        user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001
        pass

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTunion)]


INPUT_MOUSE, INPUT_KEYBOARD = 0, 1

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

WHEEL_DELTA = 120

VK = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "return": 0x0D, "shift": 0x10,
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "menu": 0x12, "pause": 0x13,
    "capslock": 0x14, "esc": 0x1B, "escape": 0x1B, "space": 0x20, "pageup": 0x21,
    "pagedown": 0x22, "end": 0x23, "home": 0x24, "left": 0x25, "up": 0x26,
    "right": 0x27, "down": 0x28, "printscreen": 0x2C, "insert": 0x2D, "delete": 0x2E,
    "del": 0x2E, "win": 0x5B, "lwin": 0x5B, "rwin": 0x5C, "apps": 0x5D,
    "numlock": 0x90, "scrolllock": 0x91, "shift_l": 0xA0, "shift_r": 0xA1,
    "ctrl_l": 0xA2, "ctrl_r": 0xA3, "alt_l": 0xA4, "alt_r": 0xA5,
    "`": 0xC0, "~": 0xC0, "-": 0xBD, "_": 0xBD, "=": 0xBB, "+": 0xBB,
    "[": 0xDB, "{": 0xDB, "]": 0xDD, "}": 0xDD, "\\": 0xDC, "|": 0xDC,
    ";": 0xBA, ":": 0xBA, "'": 0xDE, '"': 0xDE, ",": 0xBC, "<": 0xBC,
    ".": 0xBE, ">": 0xBE, "/": 0xBF, "?": 0xBF,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
}
for _c in "0123456789":
    VK[_c] = 0x30 + int(_c)
for _c in "abcdefghijklmnopqrstuvwxyz":
    VK[_c] = ord(_c.upper())

SHIFTED = {
    "~": True, "!": True, "@": True, "#": True, "$": True, "%": True, "^": True,
    "&": True, "*": True, "(": True, ")": True, "_": True, "+": True, "{": True,
    "}": True, "|": True, ":": True, '"': True, "<": True, ">": True, "?": True,
}


def _send(*inputs: INPUT) -> None:
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    if user32.SendInput(n, arr, ctypes.sizeof(INPUT)) != n:
        raise OSError(f"SendInput failed: {ctypes.get_last_error()}")


def _kbd(vk: int = 0, scan: int = 0, flags: int = 0) -> INPUT:
    return INPUT(type=INPUT_KEYBOARD, union=_INPUTunion(ki=KEYBDINPUT(0, scan, flags, 0, 0)))


def _mouse(dx: int = 0, dy: int = 0, data: int = 0, flags: int = 0) -> INPUT:
    return INPUT(type=INPUT_MOUSE, union=_INPUTunion(mi=MOUSEINPUT(dx, dy, data, flags, 0, 0)))


# ---------------------------------------------------------------- 鼠标
def cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def move_to(x: int, y: int, duration: float = 0.0) -> tuple[int, int]:
    """移动鼠标；duration>0 时线性插值分步移动（对拖拽/悬停菜单更自然）。"""
    if duration and duration > 0:
        x0, y0 = cursor_pos()
        steps = max(2, min(60, int(duration * 120)))
        for i in range(1, steps + 1):
            t = i / steps
            user32.SetCursorPos(int(x0 + (x - x0) * t), int(y0 + (y - y0) * t))
            time.sleep(duration / steps)
    user32.SetCursorPos(int(x), int(y))
    return cursor_pos()


def click(x: int | None = None, y: int | None = None, button: str = "left",
          clicks: int = 1, interval: float = 0.04) -> dict:
    if x is not None and y is not None:
        move_to(x, y)
    down, up = {
        "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
        "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
        "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
    }[button]
    for i in range(max(1, clicks)):
        _send(_mouse(flags=down), _mouse(flags=up))
        if i + 1 < clicks:
            time.sleep(interval)
    return {"ok": True, "pos": cursor_pos(), "button": button, "clicks": clicks}


def drag(x1: int, y1: int, x2: int, y2: int, duration: float = 0.5, button: str = "left") -> dict:
    down, up = {
        "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
        "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
        "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
    }[button]
    move_to(x1, y1)
    time.sleep(0.05)
    _send(_mouse(flags=down))
    time.sleep(0.05)
    move_to(x2, y2, duration=duration)
    time.sleep(0.05)
    _send(_mouse(flags=up))
    return {"ok": True, "from": [x1, y1], "to": [x2, y2]}


def scroll(amount: int, x: int | None = None, y: int | None = None) -> dict:
    """amount>0 向上滚，<0 向下滚；单位=1 个滚轮齿（120）。"""
    if x is not None and y is not None:
        move_to(x, y)
        time.sleep(0.02)
    delta = int(amount) * WHEEL_DELTA
    _send(_mouse(data=delta & 0xFFFFFFFF, flags=MOUSEEVENTF_WHEEL))
    return {"ok": True, "amount": amount, "pos": cursor_pos()}


# ---------------------------------------------------------------- 键盘
def _vk_of(token: str) -> int:
    t = token.strip().lower()
    if t in VK:
        return VK[t]
    if len(t) == 1:
        if t in VK:
            return VK[t]
        raise ValueError(f"无法映射字符按键: {token!r}")
    raise ValueError(f"未知按键: {token!r}")


def key_down(key: str) -> None:
    _send(_kbd(vk=_vk_of(key)))


def key_up(key: str) -> None:
    _send(_kbd(vk=_vk_of(key), flags=KEYEVENTF_KEYUP))


def press(combo: str, times: int = 1, interval: float = 0.03) -> dict:
    """按组合键，例如 "ctrl+s"、"alt+tab"、"enter"、"ctrl+shift+p"。"""
    parts = [p for p in combo.replace(" ", "").split("+") if p]
    if not parts:
        raise ValueError("空组合键")
    for _ in range(max(1, times)):
        for p in parts:
            key_down(p)
            time.sleep(0.012)
        for p in reversed(parts):
            key_up(p)
            time.sleep(0.012)
        time.sleep(interval)
    return {"ok": True, "combo": combo, "times": times}


def type_unicode(text: str, interval: float = 0.002) -> dict:
    """按字符逐个用 KEYEVENTF_UNICODE 注入，不依赖输入法与键盘布局。"""
    n = 0
    for ch in text:
        if ch == "\n":
            _send(_kbd(vk=VK["enter"]), _kbd(vk=VK["enter"], flags=KEYEVENTF_KEYUP))
        elif ch == "\t":
            _send(_kbd(vk=VK["tab"]), _kbd(vk=VK["tab"], flags=KEYEVENTF_KEYUP))
        else:
            units = ch.encode("utf-16-le")
            for i in range(0, len(units), 2):
                code = int.from_bytes(units[i:i + 2], "little")
                _send(_kbd(scan=code, flags=KEYEVENTF_UNICODE),
                      _kbd(scan=code, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
        n += 1
        if interval:
            time.sleep(interval)
    return {"ok": True, "chars": n, "method": "unicode"}


# ---------------------------------------------------------------- 剪贴板
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

# 64 位下必须显式声明指针型返回值，否则句柄被截断成 32 位
kernel32.GlobalAlloc.restype = ctypes.c_void_p
kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalLock.argtypes = (ctypes.c_void_p,)
kernel32.GlobalUnlock.argtypes = (ctypes.c_void_p,)
user32.OpenClipboard.argtypes = (ctypes.c_void_p,)
user32.EmptyClipboard.argtypes = ()
user32.SetClipboardData.restype = ctypes.c_void_p
user32.SetClipboardData.argtypes = (wintypes.UINT, ctypes.c_void_p)
user32.GetClipboardData.restype = ctypes.c_void_p
user32.GetClipboardData.argtypes = (wintypes.UINT,)
user32.IsClipboardFormatAvailable.argtypes = (wintypes.UINT,)
user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)


def clipboard_set(text: str) -> bool:
    data = ctypes.create_unicode_buffer(text)
    size = ctypes.sizeof(data)
    if not user32.OpenClipboard(None):
        raise OSError("OpenClipboard failed")
    try:
        user32.EmptyClipboard()
        h = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not h:
            raise OSError("GlobalAlloc failed")
        ptr = kernel32.GlobalLock(h)
        if not ptr:
            raise OSError("GlobalLock failed")
        ctypes.memmove(ctypes.c_char_p(ptr), data, size)
        kernel32.GlobalUnlock(h)
        if not user32.SetClipboardData(CF_UNICODETEXT, h):
            raise OSError("SetClipboardData failed")
    finally:
        user32.CloseClipboard()
    return True


def clipboard_get() -> str:
    if not user32.OpenClipboard(None):
        return ""
    try:
        h = user32.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return ""
        ptr = kernel32.GlobalLock(h)
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(h)
    finally:
        user32.CloseClipboard()


def paste_text(text: str, settle: float = 0.12) -> dict:
    """剪贴板 + Ctrl+V：长文本 / 中文 / 含换行代码最稳的输入方式。"""
    prev = clipboard_get()
    clipboard_set(text)
    time.sleep(settle)
    press("ctrl+v")
    time.sleep(settle)
    try:
        clipboard_set(prev)
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "chars": len(text), "method": "clipboard-paste"}


def type_text(text: str, method: str = "auto", interval: float = 0.002) -> dict:
    """method: auto | paste | unicode。auto 在长文本或含非 ASCII 时走粘贴。"""
    if method == "auto":
        method = "paste" if (len(text) > 24 or any(ord(c) > 127 for c in text) or "\n" in text) else "unicode"
    return paste_text(text) if method == "paste" else type_unicode(text, interval=interval)
