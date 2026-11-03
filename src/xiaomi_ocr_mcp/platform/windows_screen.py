"""Windows 屏幕 / 窗口操作（ctypes + Pillow，无第三方窗口库依赖）。"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from PIL import Image, ImageGrab

user32 = ctypes.WinDLL("user32", use_last_error=True)

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:  # noqa: BLE001
    pass


class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                ("rcWork", RECT), ("dwFlags", wintypes.DWORD)]


# 64 位必须声明原型：否则 HWND 被当成 32 位 int 截断，GetForegroundWindow 直接返回错误句柄
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetForegroundWindow.argtypes = ()
user32.GetWindowLongW.restype = ctypes.c_long
user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(RECT))
user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.IsWindowVisible.argtypes = (wintypes.HWND,)
user32.IsIconic.argtypes = (wintypes.HWND,)
user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
user32.BringWindowToTop.argtypes = (wintypes.HWND,)
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
user32.AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
user32.GetMonitorInfoW.argtypes = (wintypes.HMONITOR, ctypes.POINTER(MONITORINFO))
user32.EnumWindows.argtypes = (ctypes.c_void_p, wintypes.LPARAM)
user32.EnumDisplayMonitors.argtypes = (wintypes.HDC, ctypes.c_void_p, ctypes.c_void_p, wintypes.LPARAM)


def virtual_screen() -> tuple[int, int, int, int]:
    """返回虚拟桌面 (x, y, w, h)，多显示器时为整体包围盒。"""
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
    SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
    return (user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))


def list_monitors() -> list[dict]:
    out: list[dict] = []
    MonitorEnumProc = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR,
                                         wintypes.HDC, ctypes.POINTER(RECT), wintypes.LPARAM)

    def _cb(hmon, hdc, lprc, data):  # noqa: ANN001
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
        r, w = mi.rcMonitor, mi.rcWork
        out.append({
            "index": len(out),
            "x": r.left, "y": r.top,
            "width": r.right - r.left, "height": r.bottom - r.top,
            "work_x": w.left, "work_y": w.top,
            "work_width": w.right - w.left, "work_height": w.bottom - w.top,
            "primary": bool(mi.dwFlags & 1),
        })
        return 1

    user32.EnumDisplayMonitors(None, None, MonitorEnumProc(_cb), 0)
    return out


def list_windows(visible_only: bool = True, min_size: int = 40) -> list[dict]:
    res: list[dict] = []
    GWL_EXSTYLE, WS_EX_TOOLWINDOW = -20, 0x00000080
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _):  # noqa: ANN001
        if visible_only and not user32.IsWindowVisible(hwnd):
            return 1
        title = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, title, 512)
        cls = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, 256)
        if not title.value.strip():
            return 1
        ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if ex & WS_EX_TOOLWINDOW:
            return 1
        r = RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(r))
        w, h = r.right - r.left, r.bottom - r.top
        if w < min_size or h < min_size:
            return 1
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        res.append({
            "hwnd": int(hwnd), "title": title.value, "class": cls.value,
            "pid": int(pid.value), "x": r.left, "y": r.top,
            "width": w, "height": h,
            "minimized": bool(user32.IsIconic(hwnd)),
            "foreground": int(user32.GetForegroundWindow() or 0) == int(hwnd),
        })
        return 1

    user32.EnumWindows(EnumWindowsProc(_cb), 0)
    return res


def find_window(title_substring: str, exact: bool = False) -> dict | None:
    want = title_substring.strip().lower()
    for w in list_windows():
        t = w["title"].lower()
        if (t == want) if exact else (want in t):
            return w
    return None


def active_window() -> dict | None:
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    for w in list_windows():
        if w["hwnd"] == int(hwnd):
            return w
    title = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, title, 512)
    return {"hwnd": int(hwnd), "title": title.value}


def focus_window(title_substring: str, settle: float = 0.25) -> dict:
    w = find_window(title_substring)
    if w is None:
        raise LookupError(f"找不到标题含 {title_substring!r} 的窗口")
    hwnd = w["hwnd"]
    SW_RESTORE = 9
    if w["minimized"]:
        user32.ShowWindow(hwnd, SW_RESTORE)
    fg = user32.GetForegroundWindow()
    tid_cur = ctypes.windll.kernel32.GetCurrentThreadId()
    tid_fg = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    attached = False
    if tid_fg and tid_fg != tid_cur:
        attached = bool(user32.AttachThreadInput(tid_cur, tid_fg, True))
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(tid_cur, tid_fg, False)
    time.sleep(settle)
    return {"ok": True, "focused": w["title"], "hwnd": hwnd,
            "foreground_now": (active_window() or {}).get("title")}


def window_rect(title_substring: str) -> tuple[int, int, int, int] | None:
    w = find_window(title_substring)
    if w is None:
        return None
    return w["x"], w["y"], w["width"], w["height"]


def capture(region: tuple[int, int, int, int] | None = None,
            window: str | None = None,
            monitor: int | None = None,
            focus_first: bool = False,
            all_screens: bool = True) -> Image.Image:
    """抓屏。优先级：region > window > monitor > 整块虚拟桌面。

    region=(x, y, w, h)，坐标是物理像素（进程已声明 DPI 感知）。
    """
    if focus_first and window:
        focus_window(window)
    if window:
        rect = window_rect(window)
        if rect is None:
            raise LookupError(f"找不到标题含 {window!r} 的窗口")
        x, y, w, h = rect
        region = (x, y, w, h)
    elif monitor is not None:
        mons = list_monitors()
        if monitor >= len(mons):
            raise IndexError(f"显示器 index {monitor} 不存在（共 {len(mons)} 块）")
        m = mons[monitor]
        region = (m["x"], m["y"], m["width"], m["height"])
    if region is None:
        x, y, w, h = virtual_screen()
        region = (x, y, w, h)
    x, y, w, h = (int(v) for v in region)
    # 与虚拟桌面求交：窗口可能大于屏幕或部分移出屏幕，越界区域抓出来是全黑
    vx, vy, vw, vh = virtual_screen()
    x2, y2 = min(x + w, vx + vw), min(y + h, vy + vh)
    x, y = max(x, vx), max(y, vy)
    w, h = max(1, x2 - x), max(1, y2 - y)
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=all_screens)
    return img.convert("RGB")


def save(img: Image.Image, path: str, max_width: int = 0) -> str:
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    img.save(path)
    return path
