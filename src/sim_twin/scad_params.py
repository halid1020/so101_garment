"""Parse the flat ``name = expr;`` assignments of an OpenSCAD config file.

Only the subset of OpenSCAD used by ``src/platform/config.scad`` is
supported on purpose: numeric literals and arithmetic over previously
assigned names (plus ``ceil``/``floor``/``min``/``max``/``sqrt``).
Anything fancier raises immediately — better a loud failure here than a
twin silently out of sync with the printed parts.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from pathlib import Path

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_FUNCS: dict[str, Callable[..., float]] = {
    "ceil": math.ceil,
    "floor": math.floor,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}
_ASSIGN_RE = re.compile(r"^\s*(\w+)\s*=\s*(.+?)\s*;", re.MULTILINE)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _eval(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown name {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval(node.operand, env)
        return -value if isinstance(node.op, ast.USub) else value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return float(_FUNCS[node.func.id](*(_eval(a, env) for a in node.args)))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def parse_scad_params(path: Path | str) -> dict[str, float]:
    """All numeric top-level assignments of ``path``, evaluated in order.

    Later assignments win, matching OpenSCAD file-scope semantics closely
    enough for a flat config file.
    """
    text = _strip_comments(Path(path).read_text())
    env: dict[str, float] = {}
    for match in _ASSIGN_RE.finditer(text):
        name, expr = match.group(1), match.group(2)
        if name.startswith("$"):
            continue
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"{name}: cannot parse {expr!r}") from exc
        try:
            env[name] = _eval(tree, env)
        except ValueError as exc:
            raise ValueError(f"{name} = {expr!r}: {exc}") from exc
    return env
