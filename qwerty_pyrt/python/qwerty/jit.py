"""
This module is orchestrates the parsing of Python source into ASTs and
invocation of the Qwerty JIT compiler. Contains definitions of ``@qpu``
and ``@classical`` decorators. When a ``@qpu`` kernel is called, this module
jumps into the JIT'd code via ``_qwerty_harness.cpp``.
"""

import os
import ast
import weakref
import inspect
import textwrap
import functools
from types import TracebackType
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Iterable, Optional, Callable
from dataclasses import dataclass

from .err import EXCLUDE_ME_FROM_STACK_TRACE_PLEASE, _get_frame, \
                 _cook_programmer_traceback, QwertySyntaxError
#from ._qwerty_harness import set_debug, Kernel, MlirHandle, Bits, Integer, \
#                             Tuple, DebugInfo, Return, Pipe, Variable, \
#                             EmbedClassical, Kernel, Instantiate, DimVarExpr, \
#                             AST_QPU, AST_CLASSICAL, \
#                             EMBED_XOR, EMBED_SIGN, EMBED_INPLACE, \
#                             embedding_kind_name
from .runtime import dimvar, bit
from .convert_ast import AstKind, convert_ast

QWERTY_DEBUG = bool(os.environ.get('QWERTY_DEBUG', False))
QWERTY_FILE = str(os.environ.get('QWERTY_FILE', "module"))
#set_debug(QWERTY_DEBUG)
#_mlir_handle = MlirHandle(QWERTY_FILE)
#_global_generation_counter = 0

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

class KernelHandle:
    """
    An instance of this class is returned by the ``@classical`` and ``@qpu``
    decorators. Programmers can pass it around like any old Python object and
    invoke it like a Python function.
    """

    def __init__(self, ast: Kernel):
        self.ast = ast

    @_cook_programmer_traceback
    def __call__(self, *, shots=None):
        """
        Call a ``@classical`` kernel by just calling the actual Python
        function, and call a ``@qpu`` kernel by jumping into the JIT'd code.
        """

        # TODO: call OG function with actual arguments for @classical

        # TODO: return an instance of a new Histogram class that iterates over
        #       each observation instead of keys
        histo = self.ast.call(1 if shots is None else shots)

        if shots is not None:
            return histo
        else:
            # We passed shots=1. So return the only bitstring in the histogram
            bits, = histo.keys()
            return bits

def _jit(ast_kind, func, last_dimvars=None):
    """
    Obtain the Python source code for a function object ``func``, then ask the
    Python interpreter for its Python AST, then convert this Python AST to a
    Qwerty AST. If all dimension variables can be inferred from captures,
    immediately type check the AST and compile it to MLIR; otherwise, hold off
    until dimension variables are provided (see ``__getattr__()`` above).

    The initial Qwerty AST is cached between each time a Qwerty kernel is
    re-encountered with different captures or explicit dimension variables.
    """
    global QWERTY_DEBUG

    if captures is None:
        captures = []
    else:
        captures = list(captures)

    if last_dimvars is None:
        last_dimvars = []

    for i, capture in enumerate(captures):
        # Special case: Python ints need to be wrapped to simplify code
        if (qwerty_obj := _to_hybrid_obj(capture)) is not None:
            captures[i] = qwerty_obj
        elif (qwerty_tuple := _to_hybrid_tuple(capture)) is not None:
            captures[i] = qwerty_tuple
        else:
            raise QwertySyntaxError('Captures must be Qwerty types such as bit')

    capture_objs = [capture.as_qwerty_obj() for capture in captures]

    filename = inspect.getsourcefile(func) or ''
    # Minus one because we want the line offset, not the starting line
    line_offset = inspect.getsourcelines(func)[1] - 1
    func_src = inspect.getsource(func)
    # textwrap.dedent() inspired by how Triton does the same thing. compile()
    # is very unhappy if the source starts with indentation
    func_src_dedent = textwrap.dedent(func_src)
    col_offset = _calc_col_offset(func_src, func_src_dedent)

    func_ast = ast.parse(func_src_dedent)
    qwerty_ast = convert_ast(ast_kind, func_ast, filename, line_offset, col_offset)
    return KernelHandle(qwerty_ast)

class JitProxy(ABC):
    """
    The ``@qpu`` and ``@classical`` decorators are instances of this class.
    The main job of this class is to support the syntax ``@qpu[[M,N]`` by
    implementing ``__getitem__()``.
    """

    def __init__(self):
        self._last_dimvars = None

    @abstractmethod
    def _proxy_to(self, func, last_dimvars=None):
        """
        Subclasses should invoke ``_jit()`` here on ``func`` with the
        ``captures`` and ``last_dimvars`` provided and return the result.
        """
        ...

    @_cook_programmer_traceback
    def __getitem__(self, dimvars):
        """
        Support the syntax for specifying dimension variables, e.g.,
        ``@qpu[[M,N]]``.
        """
        if not isinstance(dimvars, list):
            raise QwertySyntaxError('Unknown syntax for defining kernel '
                                    'dimvars. Make sure you are using '
                                    'double brackets [[M,N]], not single '
                                    'brackets [M]')
        self._last_dimvars = dimvars
        return self

class QpuProxy(JitProxy):
    def _proxy_to(self, func, last_dimvars=None):
        return _jit(AstKind.QPU, func, last_dimvars)

class ClassicalProxy(JitProxy):
    def _proxy_to(self, func, captures=None, last_dimvars=None):
        return _jit(AstKind.CLASSICAL, func, last_dimvars)

# The infamous @qpu and @classical decorators
qpu = QpuProxy()
classical = ClassicalProxy()

__all__ = ['qpu', 'classical']
