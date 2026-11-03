"""Xiaomi-OCR-0 本地推理封装（懒加载、单例、串行化）。

模型：https://huggingface.co/xiaomi-research/Xiaomi-OCR-0
0.8B，基于 Qwen3.5-0.8B-Base，Apache-2.0。RTX 4060 8G 上 fp16 常驻约 1.9GB 显存。
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image

MODEL_DIR = Path(os.environ.get(
    "XIAOMI_OCR_MODEL_DIR",
    str(Path(__file__).parent / "models" / "Xiaomi-OCR-0"),
))

DOC_PROMPT = (
    "Extract all information from the main body of the document image and represent it in "
    "markdown format, ignoring headers and footers. Tables should be expressed in OTSL format, "
    "formulas in the document should be represented using LATEX format, and the parsing should "
    "be organized according to the reading order."
)
TEXT_PROMPT = "Extract the text in the image."
TABLE_PROMPT = "Parse the table in the image into OTSL."
FORMULA_PROMPT = "Identify the formula in the image and represent it using LATEX format."
CODE_PROMPT = (
    "Extract the source code in the image verbatim. Keep the exact indentation, punctuation, "
    "quotes, brackets and line breaks of the original code. Do not translate, do not explain, "
    "do not add line numbers. Output the code only."
)

MODE_PROMPTS = {
    "document": DOC_PROMPT,
    "text": TEXT_PROMPT,
    "table": TABLE_PROMPT,
    "formula": FORMULA_PROMPT,
    "code": CODE_PROMPT,
}

MAX_PIXELS = int(os.environ.get("XIAOMI_OCR_MAX_PIXELS", 2_600_000))


def strip_fences(text: str) -> str:
    """去掉模型可能包上的 ```lang ... ``` 围栏，返回裸代码。"""
    t = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```$", t, re.S)
    if m:
        return m.group(1)
    return t


# Q-Mask 文本锚定会吐 <loc_0><loc_500> 之类的坐标标签，语言前缀会吐 <code><_Python_>
_LOC_TAG = re.compile(r"<loc_\d+>")
_LANG_TAG = re.compile(r"</?(?:code|text|table|formula|markdown)>", re.I)
_REF_TAG = re.compile(r"<_?[A-Za-z][A-Za-z0-9_]*_>")


def clean_output(text: str, mode: str = "text") -> str:
    """剥掉模型的控制标签，只留正文。"""
    t = strip_fences(text)
    t = _LOC_TAG.sub("", t)
    t = _LANG_TAG.sub("", t)
    t = _REF_TAG.sub("", t)
    if mode == "code":
        # 行首残留的标签已清掉，去掉因此产生的空行
        t = "\n".join(line.rstrip() for line in t.split("\n")).strip("\n")
    return t.strip()


class XiaomiOCR:
    def __init__(self, model_dir: Path | str = MODEL_DIR, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        self._device_pref = device
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.dtype = None
        self._lock = threading.Lock()
        self.load_seconds = 0.0
        self.last_timing = 0.0

    # ------------------------------------------------------------ 加载
    def load(self) -> None:
        with self._lock:
            if self.model is not None:
                return
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            if not (self.model_dir / "config.json").exists():
                raise FileNotFoundError(
                    f"模型目录不存在或不完整: {self.model_dir}（先跑 python download_model.py）"
                )
            t0 = time.time()
            if self._device_pref == "cpu":
                device = "cpu"
            elif self._device_pref == "cuda" or (self._device_pref == "auto" and torch.cuda.is_available()):
                device = "cuda"
            else:
                device = "cpu"
            dtype = torch.float32 if device == "cpu" else torch.bfloat16

            self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
            self.model = (
                AutoModelForImageTextToText.from_pretrained(str(self.model_dir), dtype=dtype)
                .to(device)
                .eval()
            )
            self.device, self.dtype = device, dtype
            self.load_seconds = round(time.time() - t0, 2)

    # ------------------------------------------------------------ 推理
    def _prepare(self, image: Image.Image) -> str:
        """返回可直接塞进 chat template 的本地图片路径。超大图先降采样。"""
        img = image.convert("RGB")
        if img.width * img.height > MAX_PIXELS:
            ratio = (MAX_PIXELS / (img.width * img.height)) ** 0.5
            img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))),
                             Image.LANCZOS)
        fd, path = tempfile.mkstemp(suffix=".png", prefix="ocr_")
        os.close(fd)
        img.save(path)
        return path

    def run(self, image: Image.Image, mode: str = "text", prompt: str | None = None,
            max_new_tokens: int = 3072) -> dict:
        import torch

        self.load()
        if prompt is None:
            prompt = MODE_PROMPTS.get(mode, TEXT_PROMPT)
        path = self._prepare(image)
        t0 = time.time()
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "url": path},
                    {"type": "text", "text": prompt},
                ],
            }]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
            n_in = int(inputs["input_ids"].shape[1])
            n_out = int(new_tokens.shape[1])
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        self.last_timing = round(time.time() - t0, 2)
        return {
            "text": text.strip(),
            "mode": mode,
            "device": self.device,
            "seconds": self.last_timing,
            "prompt_tokens": n_in,
            "output_tokens": n_out,
            "tok_per_s": round(n_out / self.last_timing, 1) if self.last_timing else 0.0,
        }


_engine: XiaomiOCR | None = None
_engine_lock = threading.Lock()


def get_engine() -> XiaomiOCR:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = XiaomiOCR()
        return _engine
