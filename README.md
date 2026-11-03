# scrollback-mcp

![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![MCP](https://img.shields.io/badge/MCP-stdio-orange)

> **Unofficial community project.** Not affiliated with, endorsed by, or sponsored by Xiaomi.

**Record your IDE scrolling through a project → get the project back as files.**
CLI + MCP server: point it at a screen recording of IntelliJ IDEA and it rebuilds the
source tree (`pom.xml`, `src/main/java/...`, resources), checks the syntax, and gives you a
short list of lines worth a second look — with the video frame for each.

```
recording.mp4 ─► keyframes ─► line number + text per row ─► vote per line ─► project/ + REPORT.md
                 (screen at rest)  (gutter = exact line index)    (all frames)
```

> v0.2 is a rewrite. The 0.1 screen-control tools (click/type) are gone; this version is
> about recovering code from recordings.

## How it works

- **The gutter is the index.** Each row is keyed by the line number printed next to it, so
  stitching scrolled frames is exact, blank lines are explicit, lines nobody scrolled to are
  reported by number, and Code Vision rows ("3 个用法", author hints) are skipped for free.
- **Every row is read on its own.** Text detection misses tiny tokens (`}`, `*/`, single-digit
  line numbers), so rows are cut out by their ink and sent straight to the recogniser
  (PP-OCRv6 via RapidOCR; DirectML GPU on Windows).
- **Spaces and indentation come from pixels.** In a monospace font a gap of k cells is k
  spaces. Full-width Chinese punctuation is told apart from ASCII by its width.
- **Every frame votes**, and code-aware fixes repair typical slips (`11 注释` → `// 注释`,
  `S1f4j` → `Slf4j`, `import java.util.list` → `List`) — only where the misread form is not
  valid code, and flagged for review.
- **Names come from the code**: a Java file is named by its public type and placed by its
  `package`; IDEA's navigation bar gives the module directory and non-Java paths.

### Measured (v0.2)

Synthetic IDEA recordings rendered from a real multi-module Spring Boot project
(苍穹外卖 day01 skeleton: Java with Chinese comments, `pom.xml`, YAML, MyBatis XML), 1080p,
Darcula, Code Vision on, page-down + wheel scrolling (`tools/make_video.py`),
RTX 4060 Laptop with DirectML:

| recording | files | exact lines | time |
|---|---|---|---|
| 8 files, 12 s | 8/8, right paths | **274/274** | 20 s |
| whole project (93 files, 3197 lines), 130 s | 93/93, right paths | 98.5 %¹ | ~3.5 min |

¹ measured one iteration before the last fixes; remaining errors are mostly Chinese
punctuation and `//` markers in comments, plus a few letter-case slips.

## Recording checklist

1. **Line numbers on** (default) — they make stitching exact.
2. **Navigation bar visible** (View → Appearance → Navigation Bar, or the status-bar
   breadcrumb in the new UI) — gives each file its module and directory.
3. **Expand all folds** before scrolling: `Ctrl+Shift+NumPad +` (imports are folded by default).
4. **Inlay hints off** (Settings → Editor → Inlay Hints): parameter-name hints sit inside a line.
5. Font 14 pt or larger, soft wraps off, one editor split, native resolution.
6. **Open each file at the top and PageDown to the end**, pausing ~0.5 s per page.

## Install

```bash
pip install "scrollback-mcp[gpu,yaml] @ git+https://github.com/SorenKang/scrollback-mcp.git"
scrollback-mcp check            # ocr_backend should be dml (Windows) or cuda
scrollback-mcp check --fix-gpu  # if it says cpu although onnxruntime-directml is installed
```

Java syntax checking needs a JDK (`javac` on PATH or `JAVA_HOME`).

## CLI

```bash
scrollback-mcp rebuild recording.mp4 -o F:/replica     # resumable
scrollback-mcp report F:/replica                        # summary + review queue
scrollback-mcp crop F:/replica <file> 40 52             # PNG of those lines from the video
scrollback-mcp set  F:/replica <file> 47 "        return x;"
scrollback-mcp verify F:/replica [--build]              # javac/XML/YAML (+ mvn compile)
```

Bookkeeping lives in `F:/replica/.xocr/`: `reads.jsonl` (frame cache), `state.json`
(per-line results), `overrides.json` (your fixes — re-applied on every rebuild), `REPORT.md`.

## MCP

```json
{ "mcpServers": { "scrollback": { "command": "scrollback-mcp" } } }
```

| tool | purpose |
|---|---|
| `video_info(video)` | duration/resolution + sample frames to check readability |
| `rebuild_start(video, out)` / `rebuild_status(out)` | rebuild in the background |
| `review_queue(out)` | ranges to check: syntax errors, never-shown lines, OCR doubts |
| `view_lines(out, path, start, end)` | **the frame image** of those lines + current text |
| `set_lines(out, path, start, text)` | fix lines (kept across rebuilds) |
| `verify(out, build)` | syntax check, optional `mvn compile` |
| `read_file`, `ocr_image` | helpers |

An agent with vision closes the loop: `review_queue` → `view_lines` → compare → `set_lines`
→ `verify`, until the queue is empty.

## Testing with synthetic recordings

```bash
pip install pillow
python tools/make_video.py path/to/project -o demo.mp4 --code-vision
scrollback-mcp rebuild demo.mp4 -o out
python tools/evaluate.py path/to/project out --files demo.files.txt
```

## Limits

- Built for IntelliJ IDEA; other editors work if they show line numbers in a left gutter.
- Lines never shown on screen cannot be recovered — they are listed by number.
- Review flags are still noisy (many `single`/`cjk` flags on correct lines).
