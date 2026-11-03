# xiaomi-ocr-mcp

![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![MCP](https://img.shields.io/badge/MCP-stdio-orange)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

> **Unofficial community project.** Not affiliated with, endorsed by, or sponsored by Xiaomi.
> Uses the open-source [Xiaomi-OCR-0](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0) model (Apache-2.0).

**Screen OCR + desktop control as an MCP server.** Any MCP client can use it — Claude Desktop, Cursor, Cline, Continue, DSH, or your own agent.

Reads whatever is on screen, keeps code indentation intact, and can type back into the focused window.

```
┌──────────────┐   stdio / JSON-RPC   ┌──────────────────────────────┐
│  MCP client  │ ───────────────────► │  xiaomi-ocr-mcp              │
│  (any AI)    │ ◄─────────────────── │  ├─ fast OCR   PP-OCRv6      │
└──────────────┘                      │  ├─ vlm OCR    Xiaomi-OCR-0  │
                                      │  └─ desktop    capture/input │
                                      └──────────────────────────────┘
```

## Why two OCR engines

| Channel | Engine | One screen of code | Accuracy | Use for |
|---|---|---|---|---|
| `fast` | PP-OCRv6 + bbox rebuild | **0.23 s** (GPU) / 0.7 s (CPU) | 1.000 | plain ASCII code, logs, terminals |
| `vlm` | Xiaomi-OCR-0 (0.8B) | 6 s | 1.000 | Chinese comments, tables, formulas, multi-column |

The fast path reads **character-level bounding boxes** and rebuilds spacing and indentation from the x-coordinates — that is what makes it usable for code. It is 26× faster than the VLM and just as accurate on ASCII source.

`read_code` defaults to `engine="auto"`: it runs the fast path first and silently retries with the VLM when the result looks unreliable (high CJK ratio, unbalanced `"""`, low confidence).

The VLM channel is **optional** — it needs ~1.7 GB of weights and torch. Everything works without it.

## Install

```bash
pip install xiaomi-ocr-mcp            # fast OCR + desktop control
pip install "xiaomi-ocr-mcp[gpu]"     # + DirectML acceleration on Windows
pip install "xiaomi-ocr-mcp[vlm]"     # + Xiaomi-OCR-0 channel (torch, transformers)
```

Or run without installing:

```bash
uvx xiaomi-ocr-mcp
```

### Enable the VLM channel

```bash
pip install "xiaomi-ocr-mcp[vlm]"
xiaomi-ocr-mcp --download-model       # ~1.7 GB → ~/.cache/xiaomi-ocr-mcp/models
export XIAOMI_OCR_VLM=1
```

`--check` prints what is available on your machine.

## Configure your client

The command is the same everywhere: **`xiaomi-ocr-mcp`** (or `uvx xiaomi-ocr-mcp`).

**Claude Desktop** — `claude_desktop_config.json`

```json
{
  "mcpServers": {
    "xiaomi-ocr": { "command": "uvx", "args": ["xiaomi-ocr-mcp"] }
  }
}
```

**Cursor** — `.cursor/mcp.json`

```json
{
  "mcpServers": {
    "xiaomi-ocr": { "command": "uvx", "args": ["xiaomi-ocr-mcp"] }
  }
}
```

**Cline / Roo (VS Code)** — `cline_mcp_settings.json`

```json
{
  "mcpServers": {
    "xiaomi-ocr": {
      "command": "uvx",
      "args": ["xiaomi-ocr-mcp"],
      "disabled": false,
      "autoApprove": ["screen_info", "screenshot", "read_code", "see_screen"]
    }
  }
}
```

**Continue** — `config.yaml`

```yaml
mcpServers:
  - name: xiaomi-ocr
    command: uvx
    args: ["xiaomi-ocr-mcp"]
```

**DSH** — `cordis.patch.yml`

```yaml
- insert:
    - id: panel-mcp-xiaomi-ocr
      name: "@deepseek-ai/dsh-mcp-client"
      config:
        serverName: xiaomi-ocr
        transport: stdio
        command: xiaomi-ocr-mcp
```

## Tools

### Look

| Tool | What it does |
|---|---|
| `screen_info` | monitors, virtual desktop, active window, cursor — call this before clicking |
| `screenshot(monitor, window, region, focus_first)` | save a PNG, return its path (no OCR, fastest) |
| `read_code(monitor, window, region, engine, lang)` | **code-optimized** OCR: keeps indentation, rebuilds blank lines |
| `see_screen(monitor, window, region, mode)` | VLM OCR: `text` / `code` / `document` / `table` / `formula` |
| `ocr_image(path, mode, prompt)` | OCR a local file |
| `ocr_clipboard(mode)` | OCR whatever image is on the clipboard |

### Act

| Tool | What it does |
|---|---|
| `list_windows()` / `focus_window(title)` / `wait_for_window(title)` | find and raise windows |
| `mouse_move` / `mouse_click` / `mouse_drag` / `mouse_scroll` | pointer |
| `type_text(text, method)` | `auto` picks clipboard-paste for long/CJK/multi-line text |
| `press_keys("ctrl+s")` | key combos |
| `type_code(code)` | paste a code block verbatim |
| `clipboard_set` / `clipboard_get` | clipboard round-trip |
| `input_selftest()` | verify the synthetic-input path actually reaches the OS |
| `wait(seconds)` | sleep |

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `XIAOMI_OCR_HOME` | platform cache dir | where models and screenshots live |
| `XIAOMI_OCR_MODEL_DIR` | `$HOME/models/Xiaomi-OCR-0` | VLM weights |
| `XIAOMI_OCR_VLM` | auto (`1` if weights exist) | enable the VLM channel |
| `RAPID_CODE_GPU` | `auto` | `auto` / `dml` / `cuda` / `cpu` |
| `RAPID_CODE_MODEL` | `small` | `tiny` / `small` / `medium` |
| `XIAOMI_OCR_SHOTS` | `$HOME/shots` | screenshot output directory |

## Platform support

| | Capture + OCR | Mouse / keyboard |
|---|---|---|
| **Windows** | ✅ native (ctypes, DPI-aware) | ✅ SendInput, Unicode-safe |
| **macOS** | ✅ `screencapture` | ⚠️ `osascript`, needs Accessibility permission |
| **Linux (X11)** | ✅ `scrot` / `import` | ⚠️ `xdotool`, Wayland limited |

Windows is the reference platform — everything was measured there (RTX 4060 Laptop, DirectML).

### GPU notes

On Windows the fast path uses **DirectML** (`onnxruntime-directml`), which needs no CUDA runtime at all.

> **Gotcha:** `rapidocr` pulls in the CPU `onnxruntime`, and pip keeps both installed because
> their version numbers are unrelated. The CPU build wins at import time, so you silently stay
> on CPU. Fix it once:
>
> ```bash
> pip uninstall -y onnxruntime
> pip install --force-reinstall --no-deps onnxruntime-directml
> xiaomi-ocr-mcp --check        # fast_backend should now say "dml"
> ```

CUDA EP was tried and abandoned: onnxruntime 1.31 requires CUDA 13 + cuDNN 9, and the provider registers but never yields a device in this driver combination.

## CLI

```bash
xiaomi-ocr-mcp                  # run the MCP server on stdio
xiaomi-ocr-mcp --check          # environment diagnostics
xiaomi-ocr-mcp --selftest       # capture + OCR smoke test
xiaomi-ocr-mcp --download-model # fetch VLM weights
```

## Known limitations

- **Synthetic input can be filtered by the OS.** In sandboxed or remote sessions, `SendInput` may report success while nothing reaches other processes. Run `input_selftest` to find out; `clipboard_set` + a manual paste is the reliable fallback.
- **VLM cost scales with output length**, not image size — it is ~25 tok/s, so a full screen of code takes 6 s. That is why the fast path exists.
- **IDEA inlay hints** (`key: value`, `0 usages`) leak into OCR output. If you screen-record an IDE, post-process them away.

## License

Apache-2.0
