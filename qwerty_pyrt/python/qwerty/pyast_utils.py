"""Utilities for parsing Python ASTs"""

import ast
import inspect
import textwrap
from collections.abc import Callable

def _calc_col_offset(before_dedent, after_dedent):
    """
    Recalculate how many leading characters we removed by ``textwrap.dedent()``
    below. This way, we can give programmers an accurate column number in
    exceptions.
    """
    def first_non_ws(s: str):
        offset = len(s)
        for i, c in enumerate(s):
            if c not in ' \t':
                offset = i
                break
        return offset

    return first_non_ws(before_dedent) - first_non_ws(after_dedent)

def get_func_pyast(func: Callable[..., ...]) -> tuple[str, int, int, ast.Module]:
    """
    Extract the Python AST for a Python function ``func``. Unfortunately, this
    currently requires reading the source file and re-parsing the AST. (CPython
    intitially parses the function AST of course, but it ``free()``s it before
    even running the bytecode — our code!) A tuple of source filename, source
    line offset, source column offset, and the AST root node is returned.
    """
    filename = inspect.getsourcefile(func) or ''
    # Minus one because we want the line offset, not the starting line
    line_offset = inspect.getsourcelines(func)[1] - 1
    func_src = inspect.getsource(func)
    # textwrap.dedent() inspired by how Triton does the same thing. compile()
    # is very unhappy if the source starts with indentation
    func_src_dedent = textwrap.dedent(func_src)
    col_offset = _calc_col_offset(func_src, func_src_dedent)

    func_ast = ast.parse(func_src_dedent)
    return (filename, line_offset, col_offset, func_ast)
