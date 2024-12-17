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
from ._qwerty_harness import set_debug, Kernel, MlirHandle, Bits, Integer, \
                             Tuple, DebugInfo, Return, Pipe, Variable, \
                             EmbedClassical, Kernel, Instantiate, DimVarExpr, \
                             AST_QPU, AST_CLASSICAL, \
                             EMBED_XOR, EMBED_SIGN, EMBED_INPLACE, \
                             embedding_kind_name
from .runtime import dimvar, bit, _int, _tuple, angle, ampl, \
                     HybridPythonQwertyType
from .convert_ast import convert_ast

QWERTY_DEBUG = bool(os.environ.get('QWERTY_DEBUG', False))
QWERTY_FILE = str(os.environ.get('QWERTY_FILE', "module"))
set_debug(QWERTY_DEBUG)
_mlir_handle = MlirHandle(QWERTY_FILE)
_global_generation_counter = 0

def _dump_debug_ast(family: 'KernelFamily', ast: Kernel):
    if QWERTY_DEBUG:
        ast_filename = '{}_{}.dot'.format(family.filename_noext, ast.get_name())
        with open(ast_filename, 'w') as fp:
            fp.write(ast.dump())

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

class KernelHandle(HybridPythonQwertyType):
    """
    An instance of this class is returned by the ``@classical`` and ``@qpu``
    decorators. Programmers can pass it around like any old Python object and
    invoke it like a Python function.
    """

    def __init__(self,
                 ast: Kernel,
                 captures: Iterable[HybridPythonQwertyType],
                 family: 'KernelFamily',
                 generation: int,
                 explicit_dimvars: Optional[List[int]] = None):
        self.ast = ast
        # This is a trick to make the garbage collector not collect our callees
        # until we are also gone
        self.captures = list(captures)
        self.family = family
        self.generation = generation
        self.explicit_dimvars = explicit_dimvars
        self.embeddings = {}
        weakref.finalize(self, self._cleanup, self.ast)

    # Destructor in Python
    # https://stackoverflow.com/a/72738841/321301
    @staticmethod
    def _cleanup(ast: Kernel):
        """
        Before this object is garbage collected, notify the AST so it can erase
        its MLIR FuncOp. (Why not do this in the C++ destructor for Kernel?
        There could be multiple copies of the same Kernel. A copy being freed
        should not clear the only FuncOp.)
        """
        ast.erase()

    @_cook_programmer_traceback
    def __getitem__(self, key):
        """
        Support for syntax for setting explicit dimension variables, e.g.,
        ``kernel[[7]]`` (inside Python code)
        """
        global _mlir_handle, _global_generation_counter

        if not isinstance(key, list):
            raise QwertySyntaxError('Unknown syntax for setting kernel '
                                    'dimvars. Make sure you are using '
                                    'double brackets [[]]')
        if not key:
            raise QwertySyntaxError('Kernel dimvars is passed empty. Must have values.')
        if not all(v is Ellipsis or isinstance(v, int) for v in key):
            raise QwertySyntaxError('Value for kernel dimvars must be int.')

        explicit_dimvars = key

        if any(v < 0 for v in explicit_dimvars if v is not Ellipsis):
            raise QwertySyntaxError('Negative dimvars values not supported')

        if self.explicit_dimvars is None:
            if not self.ast.needs_explicit_dimvars():
                raise QwertySyntaxError('Explicit dimvars not needed for '
                                        'kernel {}()'.format(self.ast.get_name()))

            self.ast.register_explicit_dimvars(explicit_dimvars)
            self.explicit_dimvars = explicit_dimvars

            # If the user wrote [1,2,...] (that is, literally an ellipsis), we
            # don't want to typecheck quite yet
            if Ellipsis not in self.explicit_dimvars:
                # Dump both before and after in case type checking fails
                _dump_debug_ast(self.family, self.ast)
                self.ast.typecheck()
                _dump_debug_ast(self.family, self.ast)

                self.ast.compile(_mlir_handle)

            return self
        else:
            if self.explicit_dimvars == explicit_dimvars:
                # Short circuit here and return the existing version of ourself
                return self
            else:
                new_ast = self.family.raw_ast.copy()
                unique_gen_id = _get_unique_gen_id(new_ast.get_name(),
                                                   _global_generation_counter)
                new_ast.set_unique_gen_id(unique_gen_id)
                new_ast.infer_dimvars_from_captures(c.as_qwerty_obj()
                                                     for c in self.captures)
                new_ast.register_explicit_dimvars(explicit_dimvars)
                # Again, if the user wrote "..." in the brackets, they are
                # asking us to delay typechecking
                if Ellipsis not in explicit_dimvars:
                    # Dump both before and after in case type checking fails
                    _dump_debug_ast(self.family, new_ast)
                    new_ast.typecheck()
                    _dump_debug_ast(self.family, new_ast)

                    new_ast.compile(_mlir_handle)

                new_handle = KernelHandle(new_ast, self.captures, self.family,
                                          self.family.next_generation)
                new_handle.explicit_dimvars = explicit_dimvars
                self.family.last_handle = new_handle
                self.family.next_generation += 1
                _global_generation_counter += 1
                return new_handle

    def _embed(self, embedding_kind, operand=None):
        """
        Support ``f.sign`` inside Python for a ``@classical`` kernel ``f`` (and
        ``f.xor``/``f.inplace`` too).
        """
        global _mlir_handle, _global_generation_counter

        if (embedding_kind, operand) not in self.embeddings:
            has_freevar = self.ast.needs_explicit_dimvars()
            missing_dimvars = None
            if has_freevar:
                missing_dimvars = self.ast.missing_dimvars()

            if operand is not None:
                if not isinstance(operand, KernelHandle):
                    raise ValueError('operand to embedding must be a KernelHandle')

                if has_freevar ^ operand.ast.needs_explicit_dimvars():
                    raise QwertySyntaxError('Either both the operands to .'
                                            + embedding_kind_name(embedding_kind)
                                            + ' must have no free variable, '
                                            'or both must have a free '
                                            'variable (the same one)')

            # [0] would be this call to inspect.stack() itself
            # [1] is .sign/.xor/.inplace below, which calls self.embed(...)
            # [2] is the caller of .sign/.xor/.inplace, which is what we want
            caller_frame = inspect.stack(context=0)[2]
            dbg = DebugInfo(caller_frame.filename, caller_frame.lineno,
                            # Try to support Python <3.11
                            caller_frame.positions.col_offset+1 if hasattr(caller_frame, 'positions') else 1,
                            _get_frame())
            # Manually construct the AST for something like this:
            #     @qpu[[M,N]](k)
            #     @reversible
            #     def bubba__xor(k: cfunc[M,N], q: qubit[M]):
            #         return q | k.xor
            # If k needs explicit dimvars, then it's more like
            #     @qpu[[M,N,E]](k)
            #     @reversible
            #     def bubba__xor(k: cfunc[M,N][...], q: qubit[M]):
            #         return q | k.xor[[E]]
            wrap_instance = lambda node: \
                Instantiate(dbg.copy(), node,
                            [DimVarExpr(dv, 0) for dv in missing_dimvars]) \
                if has_freevar else node

            operand_name = 'o' if operand is not None else ''
            dummy_ret = Return(dbg.copy(),
                               Pipe(dbg.copy(),
                                    Variable(dbg.copy(), 'q'),
                                    wrap_instance(
                                        EmbedClassical(dbg.copy(), 'k',
                                                       operand_name,
                                                       embedding_kind))))
            name = self.ast.get_name() + '__' + embedding_kind_name(embedding_kind)
            type_ = self.ast.get_type().as_embedded(dbg.copy(), embedding_kind)
            dimvars = self.ast.get_dimvars() if has_freevar else []
            captures = [self]
            capture_objs = [self.as_qwerty_obj()]
            capture_names = ['k']
            capture_types = [self.ast.get_type()]
            capture_freevars = [len(missing_dimvars) if has_freevar else 0]

            if operand is not None:
                captures.append(operand)
                capture_objs.append(operand.as_qwerty_obj())
                capture_names.append('o')
                capture_types.append(operand.ast.get_type())
                capture_freevars = capture_freevars*2

            dummy_ast = Kernel(dbg.copy(),
                               AST_QPU,
                               name,
                               type_, # kernel type
                               capture_names,
                               capture_types,
                               capture_freevars, # capture free variables
                               ['q'], # argument names
                               dimvars, # type variables
                               [dummy_ret]) # body

            unique_gen_id = _get_unique_gen_id(dummy_ast.get_name(),
                                               _global_generation_counter)
            dummy_ast.set_unique_gen_id(unique_gen_id)
            dummy_ast.infer_dimvars_from_captures(capture_objs)
            if dummy_ast.needs_explicit_dimvars() ^ has_freevar:
                raise QwertySyntaxError('Embedding of kernel {}() should '
                                        'require explicit dimvars iff the '
                                        'captured kernel does. '
                                        'Compiler bug, sorry'
                                        .format(self.ast.get_name()))
            # As always, compilation is delayed if we have a free variable
            if has_freevar:
                dummy_ast.register_explicit_dimvars(self.explicit_dimvars)
            else:
                dummy_ast.typecheck()
                dummy_ast.compile(_mlir_handle)
            handle = KernelHandle(dummy_ast, captures, self.family,
                                  self.generation)
            _global_generation_counter += 1
            handle.explicit_dimvars = self.explicit_dimvars
            self.embeddings[(embedding_kind, operand)] = handle

        return self.embeddings[(embedding_kind, operand)]

    def _fixed_args(self, args):
        """
        Repartition bits to allow e.g. passing a single ``bit[N+M]`` instance
        to a ``@classical`` kernel with two args of type ``bit[N]`` and
        ``bit[M]``.
        """
        partitions = self.ast.get_type().bit_partitions()
        if not partitions or not args:
            # Conservatively fall back to passing whatever the user gave us
            return list(args)

        args_concat = functools.reduce(lambda l,r: l.concat(r), args)
        total_bits = sum(partitions)
        if total_bits != len(args_concat):
            raise TypeError('Expected a total of {} bits passed to {}(), but '
                            'only got {}'
                            .format(total_bits, self.ast.get_name(),
                                    len(args_concat)))
        fixed_args = []
        start_idx = 0
        for part in partitions:
            fixed_args.append(args_concat[start_idx:start_idx+part])
            start_idx += part
        return fixed_args

    def _fixed_result(self, ret):
        """
        Glue together tuples of bits into one giant ``bit[N+M+...]`
        """
        if isinstance(ret, tuple):
            joined_bits = 0
            total_n_bits = 0
            for b in ret:
                joined_bits = (joined_bits << b.n_bits) | int(b)
                total_n_bits += b.n_bits
            return bit[total_n_bits](joined_bits)
        else:
            return ret

    def _yield_histo(self, histo):
        """
        The ``yield`` keyword anywhere in a Python function makes the whole
        function always return a generator, so relegate the ``yield`` for
        ``__call__(histogram=True)`` to a helper function (this guy!).
        """
        for bits, count in histo.items():
            for _ in range(count):
                yield bits

    @_cook_programmer_traceback
    def __call__(self, *args, shots=None, histogram=False, acc=None):
        """
        Call a ``@classical`` kernel by just calling the actual Python
        function, and call a ``@qpu`` kernel by jumping into the JIT'd code.
        """
        global _mlir_handle

        if self.family.original_func is not None:
            if shots is not None:
                raise ValueError('shots={} keyword argument not applicable '
                                 'here'.format(shots))

            captures_and_args = self.captures + self._fixed_args(args)
            tvs_tweaked = []
            if self.explicit_dimvars is not None:
                for dv, tv_val in zip(self.ast.get_explicit_dimvar_names(), self.explicit_dimvars):
                    tv_instance = self.family.original_func.__globals__[dv]
                    tv_instance._intval = tv_val
                    tvs_tweaked.append(tv_instance)
            try:
                return self._fixed_result(self.family.original_func(*captures_and_args))
            finally:
                for tv_instance in tvs_tweaked:
                    tv_instance._intval = None

        if args:
            raise QwertySyntaxError('I do not know how to pass args into '
                                    'a kernel yet, sorry')

        if self.ast.needs_explicit_dimvars():
            # TODO: say which dimvars are needed
            raise QwertySyntaxError('Explicit dimvars needed for kernel {}()'
                                    .format(self.ast.get_name()))

        # We need to recreate the dictionary with the bit[N] type from
        # runtime.py instead of the HybridObj Bits from _qwerty_harness
        harness_histo = self.ast.call(_mlir_handle,
                                      acc,
                                      1 if shots is None else shots)
        histo = {self.from_qwerty_obj(bits): count
                 for (bits, count) in harness_histo.items()}

        if shots is not None:
            if histogram:
                return histo
            else:
                # Hack: we can't yield here directly because otherwise Python
                # decides to make this entire function return a generator, even
                # from the line above
                return self._yield_histo(histo)
        elif histogram:
            raise QwertySyntaxError('histogram= requires shots=')
        else:
            # We passed shots=1. So return the only bitstring in the histogram
            bits, = histo.keys()
            return bits

    def qasm(self, print_locs=False):
        """
        Generate OpenQASM 3.0 and return it as a ``str``.

        The default is not to print debug locations on each line (i.e.,
        ``print_locs=False``) since passing ``print_locs=True`` is very noisy.
        """
        global _mlir_handle

        if self.ast.needs_explicit_dimvars():
            # TODO: say which dimvars are needed
            raise QwertySyntaxError('Explicit dimvars needed for kernel {}()'
                                    .format(self.ast.get_name()))

        return self.ast.qasm(_mlir_handle, print_locs)

    @property
    @_cook_programmer_traceback
    def sign(self):
        """
        Support for f.sign in Python code.
        """
        return self._embed(EMBED_SIGN)

    @_cook_programmer_traceback
    def inplace(self, operand):
        """
        Support for f.inplace(f_inv) in Python code.
        """
        return self._embed(EMBED_INPLACE, operand)

    @property
    @_cook_programmer_traceback
    def xor(self):
        """
        Support for f.xor in Python code.
        """
        return self._embed(EMBED_XOR)

    def as_qwerty_obj(self):
        # Return a copy so that if this gets std::move()d away, we are not
        # hosed
        return self.ast.copy()

    # TODO: this should not be here. should probably be in types.py instead
    @classmethod
    def from_qwerty_obj(cls, qwerty_obj):
        if qwerty_obj is None:
            return None
        elif isinstance(qwerty_obj, Kernel):
            return cls(qwerty_obj)
        elif isinstance(qwerty_obj, Bits):
            return bit.from_qwerty_obj(qwerty_obj)
        elif isinstance(qwerty_obj, Integer):
            return _int.from_qwerty_obj(qwerty_obj)
        else:
            raise QwertySyntaxError('I do not know how to convert this '
                                    'qwerty_obj back to a '
                                    'HybridPythonQwertyType instance, '
                                    'sorry. This is a bug.')

@dataclass
class KernelFamily:
    """
    Tracking information for a particular Python function definition
    syntactically/physically present in the source code. These definitions may
    be re-encountered with different dimension variables, requiring them to be
    re-compiled.
    """
    # From _unique_func_id()
    func_id: int
    # Still in terms of dimvars, i.e., not typechecked
    raw_ast: Kernel
    # For debugging
    filename_noext: str
    #  @qpu[[X(37), Y,    Z(3)]]
    #         |     |      |
    #         v     v      v
    #      [True, False, True]
    tvs_has_explicit_value: List[bool]
    # Original @decorated function
    original_func: Optional[Callable[..., Any]] = None
    last_handle: Optional[KernelHandle] = None
    next_generation: int = 0

# Keys are _unique_func_id()s
_kernel_families: Dict[tuple[str, int], KernelFamily] = {}

def _unique_func_id(func):
    # To avoid unnecessarily re-parsing of the AST, and unnecessary conversion
    # to a Qwerty AST, we want to cache the untypechecked Qwerty AST for a
    # given function source code. We need a key to look up in that cache, and
    # that's what this function returns. It's tempting to say something like
    # id(func.__code__) for this, but that is only an address that could be
    # occupied by a different function once the bytecode for the original one
    # is freed. So instead use the most obvious unique identifier: the filename
    # and the line number where the function is defined. Due to Python syntax
    # (especially since we are handling only functions with decorators here),
    # this appears to be unique for the source code for a particular function.
    #
    # Note that func.__qualname__ is not enough because of
    # situations like this:
    #     def bubba(x):
    #         if x:
    #             def skippy():
    #                 return 1
    #         else:
    #             def skippy():
    #                 return 2
    #         return skippy
    #     print(bubba(True).__qualname__ == bubba(False).__qualname__) # prints True

    # TODO: don't use CPython specific APIs that may be unstable
    code = func.__code__
    return (code.co_filename, code.co_firstlineno)

def _get_unique_gen_id(name, global_generation_number):
    return '{}_{}'.format(name, global_generation_number)

def _keep_original_func(ast_kind):
    return ast_kind == AST_CLASSICAL

def _to_hybrid_obj(obj):
    if isinstance(obj, HybridPythonQwertyType):
        return obj
    elif isinstance(obj, int):
        return _int(obj)
    elif isinstance(obj, float):
        return angle(obj)
    elif isinstance(obj, complex):
        return ampl(obj)
    else:
        return None

def _to_hybrid_tuple(obj):
    try:
        it = iter(obj)
    except TypeError:
        return None # not iterable

    children = []

    while True:
        try:
            raw_obj = next(it)
        except StopIteration:
            break

        if (child_obj := _to_hybrid_obj(raw_obj)) is None:
            return None
        children.append(child_obj)

    return _tuple(children)

def _get_explicit_dimvars_from_decorator(tvs_has_explicit_value: List[bool],
                                         last_dimvars: List[int],
                                         missing_dimvars: List[str]) \
                                        -> Optional[List[int]]:
    """
    Extract ``[1,3]`` from ``@qpu[I(1),J,K(3)]``. These explicit dimension
    variables are stored in the ``@qpu`` annotation itself temporarily until we
    can fetch them here.
    """
    if not any(tvs_has_explicit_value):
        return None

    assert len(tvs_has_explicit_value) == len(last_dimvars)
    missing_dimvars = set(missing_dimvars)
    explicit_dimvars = []
    for has_explicit_value, dv in zip(tvs_has_explicit_value,
                                      last_dimvars):
        tv_name = dv._name
        if has_explicit_value:
            if tv_name not in missing_dimvars:
                raise QwertySyntaxError(
                    'Explicit dimension variable value passed with () '
                    'is unnecessary')
            explicit_dimvars.append(dv._last_explicit_val)
        elif tv_name in missing_dimvars:
            explicit_dimvars.append(...)
        else:
            # Leave this out of the list
            pass

    return explicit_dimvars

def _jit(ast_kind, func, captures=None, last_dimvars=None):
    """
    Obtain the Python source code for a function object ``func``, then ask the
    Python interpreter for its Python AST, then convert this Python AST to a
    Qwerty AST. If all dimension variables can be inferred from captures,
    immediately type check the AST and compile it to MLIR; otherwise, hold off
    until dimension variables are provided (see ``__getattr__()`` above).

    The initial Qwerty AST is cached between each time a Qwerty kernel is
    re-encountered with different captures or explicit dimension variables.
    """
    global _mlir_handle, _kernel_families, _global_generation_counter, \
           QWERTY_DEBUG

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

    func_id = _unique_func_id(func)
    if func_id not in _kernel_families:
        filename = inspect.getsourcefile(func) or ''
        # Minus one because we want the line offset, not the starting line
        line_offset = inspect.getsourcelines(func)[1] - 1
        func_src = inspect.getsource(func)
        # textwrap.dedent() inspired by how Triton does the same thing. compile()
        # is very unhappy if the source starts with indentation
        func_src_dedent = textwrap.dedent(func_src)
        col_offset = _calc_col_offset(func_src, func_src_dedent)

        func_ast = ast.parse(func_src_dedent)

        filename_noext = os.path.splitext(filename)[0]
        if QWERTY_DEBUG:
            pyast_filename = '{}_{}.pyast'.format(filename_noext, func.__name__)
            with open(pyast_filename, 'w') as fp:
                fp.write(ast.dump(func_ast, indent=4))

        raw_ast, tvs_has_explicit_value = \
            convert_ast(ast_kind, func_ast, filename, line_offset, col_offset)
        # Keep a copy of the AST with dimvars preserved (raw_ast)
        qwerty_ast = raw_ast.copy()
        qwerty_ast.infer_dimvars_from_captures(capture_objs)

        missing_dimvars = qwerty_ast.missing_dimvars()
        explicit_dimvars = _get_explicit_dimvars_from_decorator(
            tvs_has_explicit_value, last_dimvars, missing_dimvars)
        if explicit_dimvars:
            qwerty_ast.register_explicit_dimvars(explicit_dimvars)

        original_func = func if _keep_original_func(ast_kind) else None
        family = KernelFamily(func_id, raw_ast, filename_noext,
                              tvs_has_explicit_value, original_func)
        _kernel_families[func_id] = family

        unique_gen_id = _get_unique_gen_id(qwerty_ast.get_name(),
                                           _global_generation_counter)
        qwerty_ast.set_unique_gen_id(unique_gen_id)

        if not qwerty_ast.needs_explicit_dimvars():
            # Dump both before and after in case type checking fails
            _dump_debug_ast(family, qwerty_ast)
            qwerty_ast.typecheck()
            _dump_debug_ast(family, qwerty_ast)

            qwerty_ast.compile(_mlir_handle)

        handle = KernelHandle(qwerty_ast, captures, family,
                              family.next_generation, explicit_dimvars)
        family.last_handle = handle
        family.next_generation += 1
        _global_generation_counter += 1
        return handle
    else:
        family = _kernel_families[func_id]
        last_handle = family.last_handle
        if not last_handle.ast.needs_recompile(capture_objs):
            new_explicit_dimvars = _get_explicit_dimvars_from_decorator(
                family.tvs_has_explicit_value, last_dimvars,
                last_handle.ast.get_explicit_dimvar_names())
            if new_explicit_dimvars:
                return last_handle[new_explicit_dimvars]
            else:
                return last_handle
        else:
            new_ast = family.raw_ast.copy()
            unique_gen_id = _get_unique_gen_id(new_ast.get_name(),
                                               _global_generation_counter)
            new_ast.set_unique_gen_id(unique_gen_id)

            new_ast.infer_dimvars_from_captures(capture_objs)
            missing_dimvars = new_ast.missing_dimvars()
            explicit_dimvars = _get_explicit_dimvars_from_decorator(
                family.tvs_has_explicit_value, last_dimvars, missing_dimvars)
            if explicit_dimvars:
                new_ast.register_explicit_dimvars(explicit_dimvars)
            if not new_ast.needs_explicit_dimvars():
                # TODO: is there a way to avoid fully re-typechecking in this case?
                #       or is it necessary because we use captures to infer dimvars?
                new_ast.typecheck()
                new_ast.compile(_mlir_handle)
            new_handle = KernelHandle(new_ast, captures, family,
                                      family.next_generation,
                                      explicit_dimvars)
            family.last_handle = new_handle
            family.next_generation += 1
            _global_generation_counter += 1
            return new_handle

class JitProxy(ABC):
    """
    The ``@qpu`` and ``@classical`` decorators are instances of this class.
    The main job of this class is to support the syntax
    ``@qpu[[M,N](capture1, capture2)`` by implementing ``__getitem__()`` and
    ``__call__()``.
    """

    def __init__(self):
        self._last_dimvars = None

    @abstractmethod
    def _proxy_to(self, func, captures=None, last_dimvars=None):
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

    @_cook_programmer_traceback
    def __call__(self, *captures):
        """
        Support the syntax for specifying captures, e.g.,
        ``@qpu(capture1, capture2)``.
        """
        # TODO: is this a guaranteed check? Will we never capture a function?
        if len(captures) == 1 and callable(func := captures[0]) \
                and not isinstance(func, KernelHandle):
            return self._proxy_to(func, last_dimvars=self._last_dimvars)
        else:
            # Need to create a closure here @decorated with
            # @_cook_programmer_traceback since if _proxy_to throws a
            # QwertyProgrammerError the backtrace will be through this closure
            # and will not be caught by the decorator on this function
            # (__call__()). See err.py for more details.
            @_cook_programmer_traceback
            def cooked_traceback_closure(func):
                return self._proxy_to(func, captures=captures,
                                      last_dimvars=self._last_dimvars)
            return cooked_traceback_closure

class QpuProxy(JitProxy):
    def _proxy_to(self, func, captures=None, last_dimvars=None):
        return _jit(AST_QPU, func, captures, last_dimvars)

class ClassicalProxy(JitProxy):
    def _proxy_to(self, func, captures=None, last_dimvars=None):
        return _jit(AST_CLASSICAL, func, captures, last_dimvars)

# The infamous @qpu and @classical decorators
qpu = QpuProxy()
classical = ClassicalProxy()

def dump_mlir_module():
    """
    Print the Qwerty-dialect MLIR module to standard out.
    """
    global _mlir_handle
    print(_mlir_handle.dump_module_ir())

def dump_qir(to_base_profile=False):
    """
    Lower the entire program to QIR and print it to stdout. Setting
    ``to_base_profile=True`` will produce base-profile QIR [1] instead of
    unrestricted QIR intended for local JITing and simulation.

    [1]: https://github.com/qir-alliance/qir-spec/blob/main/specification/under_development/profiles/Base_Profile.md
    """
    global _mlir_handle
    print(_mlir_handle.dump_qir(to_base_profile))

def get_qir(to_base_profile=False):
    """
    Similar to ``dump_qir()`` except returning the QIR as a ``str`` instead of
    printing it out.
    """
    global _mlir_handle
    return _mlir_handle.dump_qir(to_base_profile)

def set_func_opt(do_func_opt):
    """
    Running ``set_func_opt(False)`` will disable passes that aggressively
    inline the IR. Users should never use this; it is present only for a
    portion of the evaluation for the CGO paper.
    """
    global _mlir_handle
    return _mlir_handle.set_func_opt(do_func_opt)

__all__ = ['qpu', 'classical', 'dump_mlir_module', 'dump_qir', 'get_qir', 'set_func_opt']
