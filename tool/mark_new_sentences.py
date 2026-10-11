#!/usr/bin/env python3
"""Mark, in place, every sentence of a LaTeX report that an older version lacks.

A reviewer comparing two drafts wants to see what is new without reading both.
This wraps each sentence of the current sources that does not occur in the
older version (a git ref) in ``\\bl{...}``; the report defines ``\\bl`` as a
colour. Deleted text is not shown.

The comparison is by sentence, pooled across files, so a sentence that only
moved to another section stays unmarked, and one whose number changed is
marked. Earlier rounds' colour macros (``\\new``, ``\\rev``, ``\\gr``, ``\\gn``)
and braces are ignored when comparing, so re-wrapping a sentence in a macro
does not make it new.

The marks are purely additive and the script removes its own before marking,
so it can be re-run after every edit: stripping the marks gives the input back.

Usage:
    venv/bin/python tool/mark_new_sentences.py [--base b10f99a]
        [--report documents/paper/training_analysis] [--strip]
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

MARK = r"\bl"

#: Environments whose body is never prose: skipped whole.
SKIP_ENVS = {
    "tabular",
    "tabular*",
    "tabularx",
    "tikzpicture",
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "displaymath",
    "array",
    "verbatim",
    "lstlisting",
    "thebibliography",
}
#: Floats: only the caption's text is prose.
FLOAT_ENVS = {"figure", "figure*", "table", "table*", "wrapfigure"}
HEADINGS = {"section", "subsection", "subsubsection", "paragraph", "subparagraph"}
#: Commands that end a sentence and are themselves never marked; their
#: optional and brace arguments are skipped with them.
BOUNDARY_CMDS = {
    "item",
    "label",
    "input",
    "include",
    "bibliography",
    "bibliographystyle",
    "appendix",
    "clearpage",
    "newpage",
    "maketitle",
    "centering",
    "par",
    "tableofcontents",
    "includegraphics",
    "vspace",
    "medskip",
    "bigskip",
    "smallskip",
    "noindent",
}
#: Words that a full stop follows without ending the sentence.
ABBREVIATIONS = {
    "e.g",
    "i.e",
    "al",
    "Fig",
    "Figs",
    "Eq",
    "Eqs",
    "cf",
    "vs",
    "Sec",
    "Tab",
    "approx",
    "resp",
    "No",
}
_CMD = re.compile(r"\\([A-Za-z@]+\*?|.)", re.S)
_BOUNDARY = re.compile(r"[.?!]['\")}]*\s+(?=[A-Z\\$(`])")
_COLOUR_MACROS = re.compile(r"\\(?:new|rev|gr|gn|bl)(?![A-Za-z@])")


# -- braces --------------------------------------------------------------------
def _skip_comment(text: str, i: int) -> int:
    end = text.find("\n", i)
    return len(text) if end < 0 else end


def brace_matches(text: str) -> "dict[int, int]":
    """Every unescaped brace's partner, both ways, ignoring comments."""
    out: "dict[int, int]" = {}
    stack: "list[int]" = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "%":
            i = _skip_comment(text, i)
            continue
        if c == "{":
            stack.append(i)
        elif c == "}" and stack:
            j = stack.pop()
            out[i], out[j] = j, i
        i += 1
    return out


def strip_marks(text: str) -> str:
    """Remove every ``\\bl{...}`` wrapper, keeping its contents."""
    while True:
        m = re.search(r"\\bl\{", text)
        if not m:
            return text
        match = brace_matches(text)
        close = match[m.end() - 1]
        text = text[: m.start()] + text[m.end() : close] + text[close + 1 :]


# -- regions of prose ------------------------------------------------------------
def _skip_args(text: str, i: int, match: "dict[int, int]") -> int:
    """Past any ``[...]`` and ``{...}`` arguments that start at ``i``."""
    while True:
        j = i
        while j < len(text) and text[j] in " \t":
            j += 1
        if j < len(text) and text[j] == "[":
            close = text.find("]", j)
            if close < 0:
                return i
            i = close + 1
        elif j < len(text) and text[j] == "{" and j in match:
            i = match[j] + 1
        else:
            return i


def _env_end(text: str, env: str, i: int) -> "tuple[int, int]":
    """(start, end) of the ``\\end{env}`` closing the one opened before ``i``."""
    depth = 1
    pattern = re.compile(r"\\(begin|end)\{" + re.escape(env) + r"\}")
    for m in pattern.finditer(text, i):
        depth += 1 if m.group(1) == "begin" else -1
        if depth == 0:
            return m.start(), m.end()
    return len(text), len(text)


def regions(
    text: str, start: int = 0, end: "int | None" = None, match=None
) -> "list[tuple[int, int, str]]":
    """(start, end, kind) of every run of prose, kind ``prose`` or ``heading``.

    A run never crosses a blank line, ``\\item``, ``\\begin``/``\\end``, a
    heading, ``\\label`` or a comment. A float contributes its caption only.
    """
    end = len(text) if end is None else end
    match = brace_matches(text) if match is None else match
    out: "list[tuple[int, int, str]]" = []
    seg = start
    i = start

    def close(at: int) -> None:
        if at > seg:
            out.append((seg, at, "prose"))

    while i < end:
        c = text[i]
        if c == "%":
            close(i)
            i = seg = _skip_comment(text, i)
            continue
        if c == "\n":
            j = i + 1
            while j < end and text[j] in " \t":
                j += 1
            if j < end and text[j] == "\n":
                close(i)
                while j < end and text[j] in " \t\n":
                    j += 1
                i = seg = j
                continue
        if c == "$" and text.startswith("$$", i):
            close(i)
            stop = text.find("$$", i + 2)
            i = seg = end if stop < 0 else stop + 2
            continue
        if c != "\\":
            i += 1
            continue
        m = _CMD.match(text, i)
        if m is None:  # a lone backslash at the very end
            break
        name = m.group(1)
        if name == "[":
            close(i)
            stop = text.find(r"\]", i + 2)
            i = seg = end if stop < 0 else stop + 2
        elif name in ("begin", "end"):
            close(i)
            env_m = re.match(r"\\(?:begin|end)\{([^}]*)\}", text[i:])
            env = env_m.group(1) if env_m else ""
            after = i + (env_m.end() if env_m else len(m.group(0)))
            if name == "begin" and env in SKIP_ENVS:
                after = _env_end(text, env, after)[1]
            elif name == "begin" and env in FLOAT_ENVS:
                body_end, after = _env_end(text, env, after)
                for cap in re.finditer(r"\\caption(\[[^\]]*\])?\{", text[:body_end]):
                    if cap.start() < i:
                        continue
                    open_ = cap.end() - 1
                    if open_ in match:
                        out += regions(text, open_ + 1, match[open_], match)
            elif name == "begin":
                after = _skip_args(text, after, match)
            i = seg = after
        elif name.rstrip("*") in HEADINGS:
            close(i)
            j = (
                _skip_args(text, m.end(), {})
                if text[m.end() : m.end() + 1] == "["
                else m.end()
            )
            while j < end and text[j] in " \t":
                j += 1
            if j < end and text[j] == "{" and j in match:
                out.append((j + 1, match[j], "heading"))
                i = seg = match[j] + 1
            else:
                i = seg = m.end()
        elif name in BOUNDARY_CMDS:
            close(i)
            i = seg = _skip_args(text, m.end(), match)
        else:
            i = m.end()
    close(min(i, end))
    return out


# -- sentences -------------------------------------------------------------------
def _trim(text: str, s: int, e: int) -> "tuple[int, int]":
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def sentences(text: str, match=None) -> "list[tuple[int, int, str]]":
    """(start, end, kind) of every sentence (or heading) of the prose."""
    match = brace_matches(text) if match is None else match
    out = []
    for s, e, kind in regions(text, match=match):
        if kind == "heading":
            out.append((*_trim(text, s, e), kind))
            continue
        cut = s
        for b in _BOUNDARY.finditer(text, s, e):
            word = re.search(r"([A-Za-z.]+)[.?!]$", text[s : b.start() + 1])
            if word and word.group(1).lstrip("~(") in ABBREVIATIONS:
                continue
            stop = b.end() - len(b.group(0)) + len(b.group(0).rstrip())
            out.append((*_trim(text, cut, stop), kind))
            cut = b.end()
        out.append((*_trim(text, cut, e), kind))
    return [(s, e, k) for s, e, k in out if e > s and _has_words(text[s:e])]


def _has_words(chunk: str) -> bool:
    bare = re.sub(r"\\[A-Za-z@]+\*?", " ", chunk)
    return bool(re.search(r"[A-Za-z]{2,}", bare))


def normalise(chunk: str) -> str:
    chunk = _COLOUR_MACROS.sub("", chunk)
    chunk = re.sub(r"[{}]", "", chunk).replace("~", " ")
    return re.sub(r"\s+", " ", chunk).strip()


def pool(texts: "list[str]") -> "set[str]":
    """Every sentence of these sources, normalised."""
    out: "set[str]" = set()
    for text in texts:
        text = strip_marks(text)
        out |= {normalise(text[s:e]) for s, e, _ in sentences(text)}
    return out


# -- marking ---------------------------------------------------------------------
def _pieces(
    text: str, s: int, e: int, match: "dict[int, int]"
) -> "list[tuple[int, int]]":
    """Split [s, e) into pieces whose braces balance on their own."""
    cuts: "list[tuple[int, int]]" = []  # excluded ranges
    i = s
    while i < e:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "}" and match.get(i, s) < s:
            cuts.append((i, i + 1))
        elif c == "{" and match.get(i, e) >= e:
            q = i
            # Back over a command that owns this brace and its earlier arguments.
            while q > s and text[q - 1] == "}" and match.get(q - 1, s) >= s:
                q = match[q - 1]
            name = re.search(r"\\[A-Za-z@]+\*?$", text[s:q])
            if name:
                q = s + name.start()
            cuts.append((q, i + 1))
        i += 1
    out, at = [], s
    for a, b in sorted(cuts):
        if a > at:
            out.append(_trim(text, at, a))
        at = max(at, b)
    if at < e:
        out.append(_trim(text, at, e))
    return [(a, b) for a, b in out if b > a and _has_words(text[a:b])]


def mark(text: str, old: "set[str]") -> "tuple[str, int]":
    """``text`` with every sentence not in ``old`` wrapped; and how many."""
    text = strip_marks(text)
    match = brace_matches(text)
    inserts: "list[tuple[int, str]]" = []
    count = 0
    for s, e, _ in sentences(text, match):
        if normalise(text[s:e]) in old:
            continue
        count += 1
        for a, b in _pieces(text, s, e, match):
            inserts += [(a, MARK + "{"), (b, "}")]
    for pos, token in sorted(inserts, key=lambda x: x[0], reverse=True):
        text = text[:pos] + token + text[pos:]
    return text, count


# -- the report ------------------------------------------------------------------
def report_files(report: Path) -> "list[Path]":
    return [
        report / "training_analysis.tex",
        *sorted((report / "sections").glob("*.tex")),
    ]


def body(text: str) -> "tuple[int, int]":
    """The span between ``\\begin{document}`` and ``\\end{document}``, or all."""
    b = text.find(r"\begin{document}")
    e = text.find(r"\end{document}")
    return (0 if b < 0 else b, len(text) if e < 0 else e)


def old_sources(ref: str, report: Path, repo: Path) -> "list[str]":
    rel = report.resolve().relative_to(repo.resolve())
    names = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, str(rel)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    out = []
    for name in names:
        if name.endswith(".tex") and "/tables/" not in name and "/flow/" not in name:
            src = subprocess.run(
                ["git", "show", f"{ref}:{name}"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            b, e = body(src)
            out.append(src[b:e])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base", default="b10f99a", help="git ref of the older version"
    )
    parser.add_argument(
        "--report", type=Path, default=Path("documents/paper/training_analysis")
    )
    parser.add_argument("--strip", action="store_true", help="only remove the marks")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    old = set() if args.strip else pool(old_sources(args.base, args.report, repo))
    for path in report_files(args.report):
        text = path.read_text()
        b, e = body(text)
        if args.strip:
            new, count = strip_marks(text[b:e]), 0
        else:
            new, count = mark(text[b:e], old)
        path.write_text(text[:b] + new + text[e:])
        total = len(sentences(strip_marks(text[b:e])))
        print(f"  {path.name}: {count} of {total} sentences new")


if __name__ == "__main__":
    main()
