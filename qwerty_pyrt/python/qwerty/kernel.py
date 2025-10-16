"""
This module orchestrates the parsing of Python source into ASTs and invocation
of the Qwerty JIT compiler. Contains definitions of ``@qpu`` and ``@classical``
decorators. When a ``@qpu`` kernel is called, this module jumps into the JIT'd
code.
"""

import os
import weakref
import functools
from types import TracebackType, FunctionType
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Iterable, Optional, Callable
from dataclasses import dataclass

from .err import EXCLUDE_ME_FROM_STACK_TRACE_PLEASE, \
                 _cook_programmer_traceback, get_python_vars, \
                 QwertyProgrammerError, QwertySyntaxError
from ._qwerty_pyrt import Program, QpuFunctionDef, QpuStmt, QpuExpr, \
                          EmbedKind, DimExpr, DimVar, Backend
from .runtime import dimvar, bit
from .pyast_utils import get_func_pyast
from .convert_ast import AstKind, convert_func_ast, Capturer, CaptureError, \
                         CapturedSymbol, CapturedBitReg, CapturedInt, \
                         CapturedFloat
from .prelude import PreludeHandle
from .default_qpu_prelude import default_qpu_prelude

QWERTY_DEBUG = bool(os.environ.get('QWERTY_DEBUG', False))
QWERTY_FILE = str(os.environ.get('QWERTY_FILE', "module"))
_global_func_counter = 0

# No debug info for the Program node since we don't really have a way get it
program_dbg = None
program = Program(program_dbg)

def _reset_compiler_state():
    """
    Used by unit tests to start over with a fresh program.
    """
    global program, _global_func_counter
    # TODO: also wipe _FRAME_MAP in err.py
    program = Program(program_dbg)
    _global_func_counter = 0

class KernelHandle:
    """
    An instance of this class is returned by the ``@classical`` and ``@qpu``
    decorators. Programmers can pass it around like any old Python object and
    invoke it like a Python function.
    """

    def __init__(self, func_name: str, original_func: Optional[Callable[..., Any]] = None):
        self.func_name = func_name
        self.original_func = original_func

    # TODO: add this for sign and xor too
    # TODO: hardcoding the instantiation here is a complete hack
    @property
    @_cook_programmer_traceback
    def inplace(self):
        """
        Support for f.inplace in Python code.
        """
        global program, _global_func_counter
        # TODO: set a useful debug loc
        dbg = None

        func_name = f'{self.func_name}__inplace_{_global_func_counter}'
        _global_func_counter += 1

        arg_var_expr = QpuExpr.new_variable('q', dbg)
        dim_var = DimVar.new_func_var('K', func_name)
        param = DimExpr.new_var(dim_var, dbg)
        cfunc_inst = QpuExpr.new_instantiate(self.func_name, param, dbg)
        embed_expr = QpuExpr.new_embed_classical(cfunc_inst, EmbedKind.InPlace,
                                                 dbg)
        pipe_expr = QpuExpr.new_pipe(arg_var_expr, embed_expr, dbg)
        ret_inst = QpuStmt.new_return(pipe_expr, dbg)

        args = [(None, 'q')]
        ret_type = None
        body = [ret_inst]
        is_rev = True
        dim_vars = ['K']
        func_ast = QpuFunctionDef(func_name, args, ret_type, body, is_rev,
                                  dim_vars, dbg)
        program.add_qpu_function_def(func_ast)
        return KernelHandle(func_name)

    @_cook_programmer_traceback
    def __call__(self, *args, shots=None, acc=None):
        """
        Call a ``@classical`` kernel by just calling the actual Python
        function, and call a ``@qpu`` kernel by jumping into the JIT'd code.
        """
        global program

        if self.original_func is not None:
            if shots is not None:
                raise ValueError('Cannot pass shots to Python code')

            return self.original_func(*args)
        else:
            if args:
                raise ValueError('Cannot pass arguments to quantum code')

            # TODO: return an instance of a new Histogram class that iterates
            #       over each observation instead of keys
            num_shots = 1 if shots is None else shots
            backend = Backend.from_str(acc)
            histo = dict(program.call(self.func_name, backend, num_shots,
                                      QWERTY_DEBUG))

            if shots is not None:
                return histo
            else:
                # We passed shots=1. So return the only bitstring in the
                # histogram
                bits, = histo.keys()
                return bits

    @_cook_programmer_traceback
    def qasm(self):
        return program.qasm(self.func_name, QWERTY_DEBUG)

class PyCapturer(Capturer):
    """
    A ``Capturer`` (see ``convert_ast.py``) that grabs Python variables from
    the stack frame.
    """

    def __init__(self):
        self.python_vars = get_python_vars()

    def shadows_python_variable(self, var_name: str) -> bool:
        return var_name in self.python_vars

    def capture(self, var_name: str) -> Optional[str]:
        if var_name in self.python_vars:
            python_obj = self.python_vars[var_name]
            if isinstance(python_obj, dimvar):
                # Don't try to capture dimvars. There is no useful data here in
                # Python to use in Qwerty. The binding e.g. N = dimvar('N')
                # only exists to let us say bit[N] in type annotations without
                # Python freaking out.
                return None
            elif isinstance(kernel := python_obj, KernelHandle):
                return CapturedSymbol(kernel.func_name)
            elif isinstance(bit_reg := python_obj, bit):
                return CapturedBitReg(bit_reg)
            elif isinstance(int_val := python_obj, int):
                return CapturedInt(int_val)
            elif isinstance(float_val := python_obj, float):
                return CapturedFloat(float_val)
            else:
                raise CaptureError(type(python_obj).__name__)
        else:
            return None

def _ast_kind_keeps_original_func(ast_kind: AstKind) -> bool:
    """
    Returns true if the original function should be kept and called for this
    AST kind. Currently, this is just ``@classical`` functions.
    """
    return ast_kind == AstKind.CLASSICAL

def _jit(ast_kind, func, last_dimvars=None, prelude=None):
    """
    Obtain the Python source code for a function object ``func``, then ask the
    Python interpreter for its Python AST, then convert this Python AST to a
    Qwerty AST. Return a handle by which it can be called from Python or
    another Qwerty kernel.
    """
    global QWERTY_DEBUG, _global_func_counter

    if last_dimvars is None:
        # TODO: use this?
        last_dimvars = []
    if prelude is not None:
        if not isinstance(prelude, PreludeHandle):
            raise QwertyProgrammerError(
                'Prelude passed as prelude= is not a function decorated '
                'with e.g. `@qpu_prelude`.')
        prelude = prelude._prelude

    filename, line_offset, col_offset, func_ast = get_func_pyast(func)
    name_generator = lambda ast_name: f'{ast_name}_{_global_func_counter}'
    capturer = PyCapturer()
    ast_func_def = convert_func_ast(ast_kind, func_ast, name_generator,
                                    capturer, filename, line_offset,
                                    col_offset)
    if ast_kind == AstKind.QPU:
        if prelude is not None:
            ast_func_def.add_prelude(prelude)
        program.add_qpu_function_def(ast_func_def)
    elif ast_kind == AstKind.CLASSICAL:
        if prelude is not None:
            raise QwertyProgrammerError(
                'Preludes are not supported for `@classical` functions.')
        program.add_classical_function_def(ast_func_def)
    else:
        assert False, "compiler bug: Missing handling of AstKind"

    _global_func_counter += 1
    func_name = ast_func_def.get_name()
    original_func = func if _ast_kind_keeps_original_func(ast_kind) else None
    return KernelHandle(func_name, original_func)

class JitProxy(ABC):
    """
    The ``@qpu`` and ``@classical`` decorators are instances of this class.
    The main job of this class is to support the syntax ``@qpu[[M,N]`` by
    implementing ``__getitem__()``.
    """

    def __init__(self):
        self._last_dimvars = None

    @abstractmethod
    def _proxy_to(self, func, last_dimvars=None, prelude=None):
        """
        Subclasses should invoke ``_jit()`` here on ``func`` with the
        ``captures`` and ``last_dimvars`` provided and return the result.
        """
        ...

    @abstractmethod
    def _get_default_prelude(self):
        """
        Subclasses should return the default prelude for their respective AST
        kind.
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

    def _proxy(self, func, prelude):
        """
        Call ``self._proxy_to()``. Used by ``__call__()``.
        """
        return self._proxy_to(func, last_dimvars=self._last_dimvars,
                              prelude=prelude)

    @_cook_programmer_traceback
    def __call__(self, func=None, /, *, prelude=None):
        """
        Support either Python calling the ``@qpu`` decorator on a function
        definition or someone setting the prelude with
        ``@qpu(prelude=my_prelude)``.
        """
        if func is not None:
            # The @qpu decorator is being called for a function.
            if not isinstance(func, FunctionType) \
                    or not hasattr(func, '__name__'):
                raise QwertySyntaxError('Only functions can be decorated with '
                                        '@qpu or @classical')
            return self._proxy(func, self._get_default_prelude())
        else:
            # Someone is setting @qpu(prelude=something). The next call should
            # be the decorator being applied to a function definition.
            # Need to create a closure here @decorated with
            # @_cook_programmer_traceback since if _proxy_to throws a
            # QwertyProgrammerError the backtrace will be through this closure
            # and will not be caught by the decorator on this function
            # (__call__()). See err.py for more details.
            @_cook_programmer_traceback
            def cooked_traceback_closure(func):
                return self._proxy(func, prelude)
            return cooked_traceback_closure

class QpuProxy(JitProxy):
    def _get_default_prelude(self):
        return default_qpu_prelude

    def _proxy_to(self, func, last_dimvars=None, prelude=None):
        return _jit(AstKind.QPU, func, last_dimvars, prelude)

class ClassicalProxy(JitProxy):
    def _get_default_prelude(self):
        # There is no `@classical` prelude right now
        return None

    def _proxy_to(self, func, last_dimvars=None, prelude=None):
        return _jit(AstKind.CLASSICAL, func, last_dimvars, prelude)

# The infamous @qpu and @classical decorators
qpu = QpuProxy()
classical = ClassicalProxy()

__all__ = ['qpu', 'classical']
