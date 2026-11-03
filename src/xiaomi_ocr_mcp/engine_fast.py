"""快通道：传统 OCR（RapidOCR / PP-OCRv6 ONNX）+ 基于 bbox 的缩进与空行重建。

专治「VLM 读一屏代码要 6 秒」：CUDA 上实测 **~220 ms / 屏，精度与 VLM 持平（1.0000）**。

调参结论（RTX 4060 Laptop，1100×661 代码截图，18 行）：
    small + CUDA + batch16 + side736 → 222 ms  sim 1.0000   ← 采用
    small + CUDA + batch16 + side960 → 253 ms  sim 0.9983
    medium + CUDA + batch16          → 323 ms  sim 1.0000
    tiny  + CUDA + batch16 + side960 → 199 ms  sim 0.9696
    small + CPU(24 线程)             → 1965 ms sim 1.0000   ← 线程给太多反而慢
    small + CPU(8 线程)              → 728 ms  sim 0.9974
"""

from __future__ import annotations

import os
import statistics
import threading
import time

# 必须在 import rapidocr(onnxruntime) 之前把 CUDA 12 的 DLL 目录暴露出去，
# 否则 CUDA EP 注册成功却建不出 device
from . import cuda_env

cuda_env.add_cuda_dll_dirs()

import numpy as np
from PIL import Image, ImageOps

from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

_engines: dict[str, RapidOCR] = {}
_lock = threading.Lock()

DEFAULT_MODEL = os.environ.get("RAPID_CODE_MODEL", "small").lower()
DEFAULT_LANG = os.environ.get("RAPID_CODE_LANG", "ch").lower()
# auto: 按 CUDA > DirectML > CPU 挑能用的后端；也可强制 dml / cuda / cpu
GPU_MODE = os.environ.get("RAPID_CODE_GPU", "auto").lower()

_LANGS = {
    "ch": (LangDet.CH, LangRec.CH),
    "en": (LangDet.EN, LangRec.EN),
    "latin": (LangDet.EN, LangRec.LATIN),
}
_MODELS = {
    "tiny": ModelType.TINY,
    "small": ModelType.SMALL,
    "medium": ModelType.MEDIUM,
    "mobile": ModelType.MOBILE,
    "server": ModelType.SERVER,
}


def _pick_backend() -> str:
    """返回 cuda / dml / cpu。"""
    if GPU_MODE in ("cpu", "cuda", "dml"):
        return GPU_MODE
    try:
        import onnxruntime as ort

        p = ort.get_available_providers()
        if "CUDAExecutionProvider" in p:
            return "cuda"
        if "DmlExecutionProvider" in p:
            return "dml"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _build(mt: str, lg: str, backend: str) -> RapidOCR:
    """mt 控制识别（rec）模型；检测（det）固定 tiny —— 实测 det tiny+rec small 比双 small 快 20 ms 且精度相同。"""
    det_lang, rec_lang = _LANGS[lg]
    return RapidOCR(params={
        "Global.log_level": "error",
        "Global.use_cls": False,          # 代码截图不会倒置，省掉方向分类
        "Det.model_type": ModelType.TINY,
        "Det.lang_type": det_lang,
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Det.limit_side_len": 736,
        "Rec.model_type": _MODELS[mt],
        "Rec.lang_type": rec_lang,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.rec_batch_num": 16,
        "EngineConfig.onnxruntime.use_cuda": backend == "cuda",
        "EngineConfig.onnxruntime.use_dml": backend == "dml",
        "EngineConfig.onnxruntime.intra_op_num_threads": 8,
    })


def engine(model_type: str | None = None, lang: str | None = None) -> RapidOCR:
    """model_type: tiny / small / medium；lang: ch / en / latin。

    后端按 RAPID_CODE_GPU（auto/dml/cuda/cpu）选；GPU 后端起不来时自动回退 CPU。
    """
    mt = (model_type or DEFAULT_MODEL).lower()
    lg = (lang or DEFAULT_LANG).lower()
    key = f"{mt}:{lg}"
    with _lock:
        if key in _engines:
            return _engines[key]
        backend = _pick_backend()
        eng = None
        if backend != "cpu":
            eng = _try_backend(mt, lg, backend)
        if eng is None:
            eng = _build(mt, lg, "cpu")
        _engines[key] = eng
        return eng


def _try_backend(mt: str, lg: str, backend: str, timeout: float = 25.0):
    """带超时的 GPU 后端探测：EP 建不出来时 onnxruntime 可能长时间阻塞，不能让 MCP 服务卡死。"""
    box: dict = {}

    def run() -> None:
        try:
            cand = _build(mt, lg, backend)
            cand(np.zeros((64, 256, 3), dtype=np.uint8))
            box["eng"] = cand
        except Exception:  # noqa: BLE001
            box["eng"] = None

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    return box.get("eng")


def _to_array(img: Image.Image, auto_invert: bool = True) -> np.ndarray:
    """深色主题（白字黑底）PP-OCR 认不好，按平均亮度自动反相。"""
    rgb = img.convert("RGB")
    if auto_invert:
        gray = rgb.convert("L")
        mean = sum(gray.getdata()) / (gray.width * gray.height)
        if mean < 128:
            rgb = ImageOps.invert(rgb)
    return np.array(rgb)[:, :, ::-1]  # RGB -> BGR


def _collect(res, min_score: float) -> list[dict]:
    boxes = getattr(res, "boxes", None)
    txts = getattr(res, "txts", None)
    scores = getattr(res, "scores", None)
    items: list[dict] = []
    if boxes is None or txts is None:
        return items
    for box, text, score in zip(boxes, txts, scores):
        if not text or float(score) < min_score:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        items.append({
            "x": min(xs), "y": min(ys),
            "w": max(xs) - min(xs), "h": max(ys) - min(ys),
            "cy": sum(ys) / len(ys),
            "text": text, "score": float(score),
        })
    return items


def _group_lines(items: list[dict]) -> list[list[dict]]:
    items = sorted(items, key=lambda d: (d["cy"], d["x"]))
    lines: list[list[dict]] = []
    for it in items:
        placed = False
        for line in lines:
            ref = statistics.mean(d["cy"] for d in line)
            h = max(d["h"] for d in line)
            if abs(it["cy"] - ref) <= max(6.0, h * 0.55):
                line.append(it)
                placed = True
                break
        if not placed:
            lines.append([it])
    lines.sort(key=lambda ln: statistics.mean(d["cy"] for d in ln))
    return lines


def _rebuild(items: list[dict], blank_lines: bool = True) -> str:
    """用 bbox 的 x/y 还原缩进、行内空格与空行（等宽字体下误差小于半个字符）。

    关键细节：字符宽度只能用**纯 ASCII 行**估计。中文行每字宽度约为英文的两倍，
    混进来会把 char_w 拉高，缩进整体算错（实测含中文注释时缩进全乱）。
    """
    if not items:
        return ""
    lines = _group_lines(items)

    ascii_rows = [d for d in items if d["text"] and all(ord(c) < 128 for c in d["text"])]
    est = ascii_rows or [d for d in items if d["text"]]
    widths = [d["w"] / len(d["text"]) for d in est if d["text"]]
    char_w = statistics.median(widths) if widths else 8.0
    if char_w <= 0:
        char_w = 8.0
    base_x = min(d["x"] for d in items)

    centers = [statistics.mean(d["cy"] for d in ln) for ln in lines]
    gaps = [b - a for a, b in zip(centers, centers[1:])]
    line_h = statistics.median(gaps) if gaps else 0.0

    out: list[str] = []
    for idx, line in enumerate(lines):
        line.sort(key=lambda d: d["x"])
        indent = max(0, round((line[0]["x"] - base_x) / char_w))
        parts, prev_end = [], None
        for d in line:
            if prev_end is not None:
                gap = d["x"] - prev_end
                if gap > char_w * 0.45:
                    parts.append(" " * max(1, round(gap / char_w)))
            parts.append(d["text"])
            prev_end = d["x"] + d["w"]
        if blank_lines and idx > 0 and line_h > 0:
            extra = round((centers[idx] - centers[idx - 1]) / line_h) - 1
            out.extend([""] * max(0, min(extra, 4)))
        out.append(" " * indent + "".join(parts))
    return "\n".join(out)


def _rebuild_words(word_results, blank_lines: bool = True,
                   drop_line_numbers: bool = False) -> str:
    """用**字符级**检测框重建行内空格与缩进。

    RapidOCR 的 return_word_box=True 会给出每个字符的框。等宽代码下，
    相邻字符的 x 间距除以字符宽就是空格个数——这比"整行一个框"准得多。

    两个坑（实测踩出来的）：
    1. RapidOCR 返回的行分组跟着 det 框走，一个视觉行可能被拆成好几组，
       所以必须先扁平化再自己按 y 聚类；
    2. 中文字符之间天然有间距，插空格会把「规则项的值」拆成「规则项 的值」，
       只对两侧都是 ASCII 的间隙插空格。
    """
    # 1) 扁平化
    chars: list[dict] = []
    for line in word_results:
        for item in line:
            ch, score, box = item[0], item[1], item[2]
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            if not ch:
                continue
            chars.append({
                "ch": ch, "score": float(score),
                "x0": min(xs), "x1": max(xs),
                "y0": min(ys), "y1": max(ys),
                "cy": sum(ys) / len(ys),
                "h": max(ys) - min(ys),
                "w": max(xs) - min(xs),
            })
    if not chars:
        return ""

    heights = [c["h"] for c in chars if c["h"] > 0]
    line_h = statistics.median(heights) if heights else 20.0

    # 2) 按 y 聚行（以字符高度为容差）
    chars.sort(key=lambda c: (c["cy"], c["x0"]))
    rows: list[list[dict]] = []
    for c in chars:
        placed = False
        for row in rows:
            ref = statistics.mean(x["cy"] for x in row)
            if abs(c["cy"] - ref) <= line_h * 0.55:
                row.append(c)
                placed = True
                break
        if not placed:
            rows.append([c])
    rows.sort(key=lambda r: statistics.mean(x["cy"] for x in r))

    # 3) 字符宽：取 ASCII 窄字符的中位数
    widths = [c["w"] for c in chars if c["w"] > 0 and c["ch"].isascii() and c["ch"].strip()]
    char_w = statistics.median(widths) if widths else 8.0
    if char_w <= 0:
        char_w = 8.0

    # 3.5) 剔除行号列（IDEA 行号 1~3 位数字，与代码之间有明显间隙）
    if drop_line_numbers:
        kept: list[list[dict]] = []
        for row in rows:
            row.sort(key=lambda c: c["x0"])
            k = 0
            while k < len(row) and k < 3 and row[k]["ch"].isdigit():
                k += 1
            if k and k < len(row) and (row[k]["x0"] - row[k - 1]["x1"]) > char_w * 1.5:
                del row[:k]
            if not row:
                continue
            # 整行只剩数字/杂符的（空行的行号），整行丢掉
            text_only = "".join(x["ch"] for x in row).strip()
            if text_only and len(text_only) <= 3 and not any(ch.isalpha() for ch in text_only):
                continue
            kept.append(row)
        rows = kept

    base_x = min(r[0]["x0"] for r in rows if r)
    centers = [statistics.mean(c["cy"] for c in r) for r in rows]
    gaps = [b - a for a, b in zip(centers, centers[1:])]
    row_h = statistics.median(gaps) if gaps else line_h * 1.5

    out: list[str] = []
    for idx, row in enumerate(rows):
        if not row:
            continue
        row.sort(key=lambda c: c["x0"])
        indent = max(0, round((row[0]["x0"] - base_x) / char_w))
        parts: list[str] = []
        for i, c in enumerate(row):
            if i > 0:
                prev = row[i - 1]
                gap = c["x0"] - prev["x1"]
                both_ascii = prev["ch"].isascii() and c["ch"].isascii()
                cross = prev["ch"].isascii() != c["ch"].isascii()
                need = False
                if (both_ascii or cross) and prev["ch"].strip() and c["ch"].strip():
                    thresh = 0.55 if both_ascii else 0.62
                    if gap > char_w * thresh:
                        # 「标签ID」「租户ID」在真实代码里 129 次全部紧贴，例外放行
                        tail = "".join(x["ch"] for x in row[i:i + 2])
                        need = not tail.startswith("ID")
                if need:
                    parts.append(" " * min(max(1, round(gap / char_w)), 4))
            parts.append(c["ch"])
        text = "".join(parts)
        # 过滤噪声行：单字符、且不是常见代码符号
        if len(text.strip()) == 1 and not text.strip().isascii():
            continue
        if blank_lines and out and idx > 0 and row_h > 0:
            extra = round((centers[idx] - centers[idx - 1]) / row_h) - 1
            out.extend([""] * max(0, min(extra, 4)))
        out.append(" " * indent + text)
    return "\n".join(out)


def ocr_code(img: Image.Image, model_type: str | None = None, lang: str | None = None,
             min_score: float = 0.5, blank_lines: bool = True, auto_invert: bool = True,
             word_level: bool = True, drop_line_numbers: bool = False) -> dict:
    t0 = time.time()
    arr = _to_array(img, auto_invert=auto_invert)
    eng = engine(model_type, lang)
    res = eng(arr, return_word_box=True) if word_level else eng(arr)

    word_results = getattr(res, "word_results", None) if word_level else None
    if word_results:
        code = _rebuild_words(word_results, blank_lines=blank_lines,
                              drop_line_numbers=drop_line_numbers)
        scores = [float(it[1]) for line in word_results for it in line]
        boxes = getattr(res, "boxes", None)
        items = len(boxes) if boxes is not None else 0
    else:
        items_list = _collect(res, min_score)
        code = _rebuild(items_list, blank_lines=blank_lines)
        scores = [d["score"] for d in items_list]
        items = len(items_list)

    cjk = sum(1 for ch in code if "\u4e00" <= ch <= "\u9fff")
    return {
        "code": code,
        "items": items,
        "lines": code.count("\n") + 1 if code else 0,
        "avg_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "min_score": round(min(scores), 4) if scores else 0.0,
        "cjk_ratio": round(cjk / len(code), 4) if code else 0.0,
        "quote_unbalanced": code.count('"""') % 2 == 1 or code.count("'''") % 2 == 1,
        "seconds": round(time.time() - t0, 3),
    }

