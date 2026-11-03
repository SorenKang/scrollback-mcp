r"""把 CUDA 运行时 DLL 暴露给 onnxruntime-gpu。

关键事实（实测踩出来的）：
    onnxruntime-gpu 1.31.1 要求 **CUDA 13.\*** + **cuDNN 9.\***（不是 CUDA 12！）
    报错原文：`Require cuDNN 9.* and CUDA 13.*, and the latest MSVC runtime.`

torch 2.14.0+cu130 的 `torch/lib` 里正好带齐了这套（cudart64_13 / cudnn*_9 / cublas64_13），
直接把它 add_dll_directory 进去就行，不用另外装 nvidia-cuda-*-cu13 的 pip 包。
只有在没有 torch 的环境里才回退去扫 `site-packages/nvidia/*/bin`。

注意：add_dll_directory 返回的句柄必须持有，否则被 GC 回收后目录就失效了。
"""

from __future__ import annotations

import glob
import os
import site
import sys

_HANDLES: list = []
DONE = False
DIRS: list[str] = []


def add_cuda_dll_dirs() -> list[str]:
    global DONE, DIRS
    if DONE:
        return DIRS
    DONE = True
    if sys.platform != "win32":
        return DIRS

    # 只用 pip 装的 nvidia CUDA 12 运行时。
    # 注意：**不能**把 torch/lib 也加进来 —— 那里面的 cudnn64_9.dll 是给 CUDA 13 编的，
    # 和 nvidia 包里 CUDA 12 版的同名 DLL 冲突，先加载的赢，会让 onnxruntime 的 CUDA EP 建不出设备。
    roots: list[str] = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:  # noqa: BLE001
        pass
    try:
        roots.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001
        pass
    for sp in roots:
        nv = os.path.join(sp, "nvidia")
        if not os.path.isdir(nv):
            continue
        for pat in ("*/bin", "*/lib"):
            for p in sorted(glob.glob(os.path.join(nv, pat))):
                try:
                    _HANDLES.append(os.add_dll_directory(p))
                    DIRS.append(p)
                except OSError:
                    pass
    return DIRS


def preload() -> dict:
    """加载 CUDA DLL 并让 onnxruntime 预热（1.21+ 提供 preload_dlls）。"""
    dirs = add_cuda_dll_dirs()
    info: dict = {"dll_dirs": dirs, "preloaded": False, "providers": []}
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
                info["preloaded"] = True
            except Exception as exc:  # noqa: BLE001
                info["preload_error"] = f"{type(exc).__name__}: {exc}"
        info["providers"] = ort.get_available_providers()
    except Exception as exc:  # noqa: BLE001
        info["import_error"] = f"{type(exc).__name__}: {exc}"
    return info

