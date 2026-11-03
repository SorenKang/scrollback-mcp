"""Check a rebuilt tree: Java syntax (javac, parse phase only), XML and YAML well-formedness,
and optionally a real Maven build."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .rebuild import WORK, work_dir

_JAVAC_ERR = re.compile(r"^(.*\.java):(\d+): error: (.*)$")


def _files(out: Path, *exts: str) -> list[Path]:
    return sorted(p for p in out.rglob("*") if p.is_file() and p.suffix in exts
                  and WORK not in p.relative_to(out).parts)


def find_javac() -> str | None:
    home = os.environ.get("JAVA_HOME")
    if home:
        for name in ("javac.exe", "javac"):
            p = Path(home) / "bin" / name
            if p.exists():
                return str(p)
    return shutil.which("javac")


def find_mvn() -> str | None:
    hit = shutil.which("mvn") or shutil.which("mvn.cmd")
    if hit:
        return hit
    # IntelliJ IDEA ships a Maven distribution
    for base in (Path("C:/Program Files/JetBrains"), Path("D:/"), Path("C:/")):
        if base.exists():
            for p in base.glob("*/*IDEA*/plugins/maven*/lib/maven3/bin/mvn.cmd"):
                return str(p)
            for p in base.glob("*IDEA*/plugins/maven*/lib/maven3/bin/mvn.cmd"):
                return str(p)
    return None


def java_syntax(out: Path) -> dict[str, list[dict]]:
    files = _files(out, ".java")
    javac = find_javac()
    if not files or not javac:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        argfile = Path(tmp) / "files.txt"
        argfile.write_text("\n".join(f'"{f.as_posix()}"' for f in files), encoding="utf-8")
        # stop after parsing: unresolved imports/symbols (no dependencies here) are not our problem
        p = subprocess.run(
            [javac, "-J-Duser.language=en", "-J-Duser.country=US", "-encoding", "UTF-8",
             "-XDshould-stop.ifError=PARSE", "-XDshould-stop.ifNoError=PARSE",
             "-proc:none", "-Xmaxerrs", "10000", "-d", tmp, f"@{argfile}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
    errs: dict[str, list[dict]] = {}
    for line in p.stderr.splitlines():
        m = _JAVAC_ERR.match(line)
        if m:
            rel = Path(m.group(1)).resolve().relative_to(out.resolve()).as_posix()
            errs.setdefault(rel, []).append({"line": int(m.group(2)), "msg": m.group(3)})
    return errs


def xml_wellformed(out: Path) -> dict[str, list[dict]]:
    import xml.etree.ElementTree as ET

    errs: dict[str, list[dict]] = {}
    for f in _files(out, ".xml"):
        try:
            ET.parse(f)
        except ET.ParseError as e:
            line = e.position[0] if getattr(e, "position", None) else 0
            errs.setdefault(f.relative_to(out).as_posix(), []).append({"line": line, "msg": str(e)})
    return errs


def yaml_wellformed(out: Path) -> dict[str, list[dict]]:
    try:
        import yaml
    except ImportError:
        return {}
    errs: dict[str, list[dict]] = {}
    for f in _files(out, ".yml", ".yaml"):
        try:
            list(yaml.safe_load_all(f.read_text(encoding="utf-8")))
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            errs.setdefault(f.relative_to(out).as_posix(), []).append(
                {"line": mark.line + 1 if mark else 0, "msg": str(e).splitlines()[0]})
    return errs


def maven_build(out: Path, timeout: int = 900) -> dict:
    mvn = find_mvn()
    if not mvn or not (out / "pom.xml").exists():
        return {"ran": False, "reason": "no mvn found" if not mvn else "no pom.xml at project root"}
    p = subprocess.run([mvn, "-q", "-B", "-DskipTests", "compile"], cwd=out, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
    tail = "\n".join((p.stdout + p.stderr).splitlines()[-60:])
    return {"ran": True, "ok": p.returncode == 0, "output_tail": tail}


def check(out: str | Path, build: bool = False) -> dict:
    out = Path(out)
    errs: dict[str, list[dict]] = {}
    for part in (java_syntax(out), xml_wellformed(out), yaml_wellformed(out)):
        for k, v in part.items():
            errs.setdefault(k, []).extend(v)
    res: dict = {"javac": bool(find_javac()), "files_with_errors": len(errs),
                 "errors": sum(len(v) for v in errs.values()), "by_file": errs}
    if build:
        res["maven"] = maven_build(out)
    (work_dir(out) / "verify.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    return res
