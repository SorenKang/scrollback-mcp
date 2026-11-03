"""Conservative, language-aware repairs of typical recogniser slips in source code.

`fix_line` runs on every reading (any file type). `fix_file` runs once per rebuilt file and
knows the language; it only touches *code* (never string literals or comments, where any
text is legal) and only where the misread form is not valid code to begin with. Lines it
changes are flagged `autofix` so they still show up in the review queue.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- every file
# An identifier never starts with a digit: `<1ombok>` / `1ist` are a misread `l`.
_LEADING_ONE = re.compile(r"(?<![\w.$])1(?=[a-z]{2,}\b)")
# `http: //host` — the recogniser likes a space after the scheme colon
_URL_SPACE = re.compile(r"\b(https?|ftp|file|jdbc:\w+):\s+//")
# A lone `//` (comment marker followed by a space) is read as `11`, `/1`, `ll`…  Rewritten
# only where a statement cannot go on like that: at line start or after `;{})`, followed by
# Chinese text or by plain words.
_SLASHES = r"(?:11|1/|/1|l/|/l|ll|1l|l1|I/|/I|II|//)"
_COMMENT = re.compile(
    r"(^|[;{})]\s+)" + _SLASHES
    + r"(?=\s+(?:[^\x00-\x7f]|[A-Za-z]+\s+[A-Za-z]|[A-Za-z]+\s*$))")

# Well-known names that contain letters the recogniser swaps with digits (l↔1, o↔0).
# A token that only matches one of these after folding those look-alikes is the known name.
_KNOWN = """Slf4j slf4j Log4j log4j mysql MySQL lombok Lombok xml yml yaml html http https url
null util utils Util Utils lang Long long List list Map Set Collection Collections Optional
Objects Bool Boolean Double Float Object Serializable Override Autowired Resource Value
Component Service Controller RestController RequestMapping GetMapping PostMapping PutMapping
DeleteMapping RequestBody RequestParam PathVariable ResponseBody Mapper Select Insert Update
Delete Param Data Builder NoArgsConstructor AllArgsConstructor Getter Setter ToString
EqualsAndHashCode Transactional Configuration Bean Slf4j SpringBootApplication
EnableTransactionManagement LocalDate LocalDateTime LocalTime BigDecimal ArrayList HashMap
HashSet LinkedList LinkedHashMap TreeMap Collectors Stream Arrays Date Calendar Locale UUID
Random Iterator Integer String Character Exception RuntimeException IllegalArgumentException
IllegalStateException NullPointerException IOException Logger LoggerFactory Log JSON JSONObject
JSONArray Json localhost true false public private protected static final void class
interface enum extends implements import package return new this super if else for while
do switch case default break continue try catch finally throw throws synchronized volatile
transient abstract native strictfp instanceof boolean char byte short int float double
config Config application spring springframework mybatis baomidou alibaba druid aliyun oss
OSS redis Redis jwt Jwt JWT token Token login Login admin Admin employee Employee
""".split()
_FOLD = str.maketrans({"1": "l", "I": "l", "|": "l", "0": "o", "O": "o"})
_KNOWN_BY_FOLD: dict[str, list[str]] = {}
for _w in _KNOWN:
    _KNOWN_BY_FOLD.setdefault(_w.translate(_FOLD).lower(), []).append(_w)
_TOKEN = re.compile(r"[A-Za-z0-9_$]+")


def _known_fix(tok: str) -> str:
    """`S1f4j` → `Slf4j`, `mysq1` → `mysql`: only tokens with a digit glued to letters."""
    if not re.search(r"[A-Za-z][01]|[01][A-Za-z]", tok):
        return tok
    cands = _KNOWN_BY_FOLD.get(tok.translate(_FOLD).lower())
    if not cands:
        return tok
    # keep the case pattern the recogniser saw where it is unambiguous
    same_case = [c for c in cands if c[0].isupper() == tok[0].isupper()]
    return (same_case or cands)[0]


def fix_line(text: str) -> str:
    text = _LEADING_ONE.sub("l", text)
    text = _URL_SPACE.sub(r"\1://", text)
    text = _COMMENT.sub(r"\1//", text)
    text = _TOKEN.sub(lambda m: _known_fix(m.group(0)), text)
    return text


# ---------------------------------------------------------------- Java
# JDK / common library classes: `import java.util.list;` is `List`
_CLASSES = {c.lower(): c for c in """List ArrayList LinkedList Map HashMap LinkedHashMap TreeMap
Set HashSet LinkedHashSet TreeSet Collection Collections Arrays Iterator Optional Objects Date
Calendar Locale UUID Random Scanner Properties Stream Collectors IntStream LocalDate
LocalDateTime LocalTime Instant Duration ZoneId DateTimeFormatter BigDecimal BigInteger
Serializable File Files Path Paths InputStream OutputStream FileInputStream FileOutputStream
ByteArrayInputStream ByteArrayOutputStream IOException Reader Writer BufferedReader
StandardCharsets Base64 Pattern Matcher TimeUnit Executors ExecutorService
ConcurrentHashMap AtomicInteger AtomicLong Function Supplier Consumer Predicate
HttpServletRequest HttpServletResponse""".split()}

# `Object 0 : list` — a variable cannot be a number: after a type name, `0` is `o`
_DIGIT_VAR = re.compile(r"(\b\w+|[>\]])(\s+)0(?=\s*[:=;,)])")
# `millis() 1 1000` — two operands side by side is not Java: the `1` is a `/`
_SLASH_OP = re.compile(r"(\b\w+|[)\]])(\s+)[1lI|](?=\s+[\w(\"'])")
_PRIMITIVES = {"int", "long", "short", "byte", "char", "boolean", "double", "float", "var"}
_KEYWORDS = {"return", "case", "throw", "new", "else", "yield", "assert", "default", "in",
             "instanceof", "extends", "implements", "import", "package", "goto"}
_IMPORT = re.compile(r"^(\s*import\s+(?:static\s+)?[\w.]+\.)([a-z]\w*)(\s*;)")


def _java_code_spans(line: str, in_block: bool) -> tuple[list[tuple[int, int]], bool]:
    """[start, end) ranges of `line` that are code (not comment, not string/char literal)."""
    spans, i, start, n = [], 0, 0, len(line)
    if in_block:
        end = line.find("*/")
        if end < 0:
            return [], True
        i = start = end + 2
    while i < n:
        c = line[i]
        if line.startswith("//", i):
            break
        if line.startswith("/*", i):
            spans.append((start, i))
            end = line.find("*/", i + 2)
            if end < 0:
                return spans, True
            i = start = end + 2
            continue
        if c in "\"'":
            spans.append((start, i))
            j = i + 1
            while j < n and line[j] != c:
                j += 2 if line[j] == "\\" else 1
            i = start = j + 1
            continue
        i += 1
    spans.append((start, min(i, n)))
    return [(a, b) for a, b in spans if b > a], False


def _fix_java_code(code: str) -> str:
    def var(m: re.Match) -> str:
        left = m.group(1)
        typ = left in (">", "]") or left in _PRIMITIVES or (left[0].isupper() and left.isidentifier())
        return f"{left}{m.group(2)}o" if typ else m.group(0)

    def slash(m: re.Match) -> str:
        left = m.group(1)
        if left in _KEYWORDS or left in _PRIMITIVES or (left[0].isupper() and left.isidentifier()):
            return m.group(0)          # `int l = …`, `Long l : …` — a real variable named l/I
        return f"{left}{m.group(2)}/"

    code = _DIGIT_VAR.sub(var, code)
    code = _SLASH_OP.sub(slash, code)
    return code


def fix_file(lines: list[str], ext: str) -> list[tuple[str, bool]]:
    """Language-aware fixes for a whole file. Returns (text, changed) per line."""
    if ext != ".java":
        return [(ln, False) for ln in lines]
    out = []
    in_block = False
    for ln in lines:
        new = ln
        m = _IMPORT.match(new)
        if m and m.group(2).lower() in _CLASSES:
            new = m.group(1) + _CLASSES[m.group(2).lower()] + m.group(3)
        # Javadoc / block comment bodies are prose
        if not in_block and not new.lstrip().startswith(("*", "/*")):
            spans, in_block = _java_code_spans(new, False)
            parts, last = [], 0
            for a, b in spans:
                parts.append(new[last:a])
                parts.append(_fix_java_code(new[a:b]))
                last = b
            parts.append(new[last:])
            new = "".join(parts)
        else:
            _, in_block = _java_code_spans(new, in_block or new.lstrip().startswith("/*"))
        out.append((new, new != ln))
    return out
