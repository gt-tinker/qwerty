"""
Convert a Python AST to a Qwerty AST by recognizing patterns in the Python AST
formed by Qwerty syntax.
"""

import ast
import math
import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import List, Tuple, Union, Optional
from .runtime import bit
from .err import EXCLUDE_ME_FROM_STACK_TRACE_PLEASE, QwertySyntaxError, \
                 get_frame, set_dbg_frame
from ._qwerty_pyrt import DebugLoc, RegKind, Type, QpuFunctionDef, \
                          QpuPrelude, ClassicalFunctionDef, FloatExpr, \
                          Vector, Basis, QpuStmt, ClassicalStmt, QpuExpr, \
                          ClassicalExpr, BasisGenerator, UnaryOpKind, \
                          BinaryOpKind, EmbedKind, DimVar, DimExpr, \
                          ExprMacroPattern, BasisMacroPattern, RecDefParam

#################### COMMON CODE FOR BOTH @QPU AND @CLASSICAL DSLs ####################

class AstKind(Enum):
    QPU = 1
    CLASSICAL = 2

class CapturedValue(ABC):
    """A captured value."""

    @abstractmethod
    def to_symbol_name(self, dbg: Optional[DebugLoc]) -> str:
        """
        Return this value as a symbol name or throw a ``QwertySyntaxError`` if
        that is not possible.
        """
        ...

    @abstractmethod
    def to_expr(self, expr_class: type[QpuExpr | ClassicalExpr],
                dbg: Optional[DebugLoc]) -> QpuExpr | ClassicalExpr:
        """
        Return this value as an expression AST node or throw a
        ``QwertySyntaxError`` if that is not possible.
        """
        ...

    @abstractmethod
    def to_dim_expr(self, dbg: Optional[DebugLoc]) -> DimExpr:
        """
        Return this value as a dimension variable expression or throw a
        ``QwertySyntaxError`` if that is not possible.
        """
        ...

    @abstractmethod
    def to_float_expr(self, dbg: Optional[DebugLoc]) -> FloatExpr:
        """
        Return this value as a float expression or throw a
        ``QwertySyntaxError`` if that is not possible.
        """
        ...

class CapturedSymbol(CapturedValue):
    """A captured (and mangled) function name."""

    def __init__(self, mangled_name: str):
        self.mangled_name = mangled_name

    def to_symbol_name(self, dbg: Optional[DebugLoc]) -> str:
        return self.mangled_name

    def to_expr(self, expr_class: type[QpuExpr | ClassicalExpr],
                dbg: Optional[DebugLoc]) -> QpuExpr | ClassicalExpr:
        return expr_class.new_variable(self.mangled_name, dbg)

    def to_dim_expr(self, dbg: Optional[DebugLoc]) -> DimExpr:
        raise QwertySyntaxError("Functions cannot be used in dimension "
                                "variable expressions", dbg)

    def to_float_expr(self, dbg: Optional[DebugLoc]) -> FloatExpr:
        raise QwertySyntaxError("Functions cannot be used in float "
                                "expressions", dbg)

class CapturedBitReg(CapturedValue):
    """A captured ``bit[N]``."""

    def __init__(self, bit_reg: bit):
        self.bit_reg = bit_reg

    def to_symbol_name(self, dbg: Optional[DebugLoc]) -> str:
        raise QwertySyntaxError("Bit registers cannot be used as symbol names",
                                dbg)

    def to_expr(self, expr_class: type[QpuExpr | ClassicalExpr],
                dbg: Optional[DebugLoc]) -> QpuExpr | ClassicalExpr:
        return expr_class.new_bit_literal(int(self.bit_reg),
                                          self.bit_reg.n_bits,
                                          dbg)

    def to_dim_expr(self, dbg: Optional[DebugLoc]) -> DimExpr:
        raise QwertySyntaxError("Bit registers cannot be used in dimension "
                                "variable expressions", dbg)

    def to_float_expr(self, dbg: Optional[DebugLoc]) -> FloatExpr:
        raise QwertySyntaxError("Bit registers cannot be used in float "
                                "expressions", dbg)

class CapturedInt(CapturedValue):
    """A captured ``int``."""

    def __init__(self, int_val: int):
        self.int_val = int_val

    def to_symbol_name(self, dbg: Optional[DebugLoc]) -> str:
        raise QwertySyntaxError("ints cannot be used as symbol names", dbg)

    def to_expr(self, expr_class: type[QpuExpr | ClassicalExpr],
                dbg: Optional[DebugLoc]) -> QpuExpr | ClassicalExpr:
        raise QwertySyntaxError("ints cannot be captured in Qwerty "
                                "kernels", dbg)

    def to_dim_expr(self, dbg: Optional[DebugLoc]) -> DimExpr:
        return DimExpr.new_const(self.int_val, dbg)

    def to_float_expr(self, dbg: Optional[DebugLoc]) -> FloatExpr:
        return FloatExpr.new_dim_expr(self.to_dim_expr(), dbg)

class CapturedFloat(CapturedValue):
    """A captured ``float``."""

    def __init__(self, float_val: float):
        self.float_val = float_val

    def to_symbol_name(self, dbg: Optional[DebugLoc]) -> str:
        raise QwertySyntaxError("floats cannot be used as symbol names", dbg)

    def to_expr(self, expr_class: type[QpuExpr | ClassicalExpr],
                dbg: Optional[DebugLoc]) -> QpuExpr | ClassicalExpr:
        raise QwertySyntaxError("floats cannot be captured in Qwerty "
                                "kernels", dbg)

    def to_dim_expr(self, dbg: Optional[DebugLoc]) -> DimExpr:
        raise QwertySyntaxError("floats cannot be used in dimension "
                                "variable expressions", dbg)

    def to_float_expr(self, dbg: Optional[DebugLoc]) -> FloatExpr:
        return FloatExpr.new_const(self.float_val, dbg)

class CaptureError(Exception):
    """
    Represents a failure to capture a Python object. This exception exists
    because we want to be noisy about that case, not silently fail.
    """

    def __init__(self, type_name):
        self.type_name = type_name

class Capturer(ABC):
    """In charge of capturing Python variables inside Qwerty kernels."""

    @abstractmethod
    def shadows_python_variable(self, var_name: str) -> bool:
        """
        Returns true if this variable name would shadow a Python variable name.
        """
        ...

    @abstractmethod
    def capture(self, var_name: str) -> Optional[CapturedValue]:
        """
        If ``var_name`` is a Python variable, capture it and return the mangled
        name (or bit instance if a bit register was captured). If it is not a
        Python variable, return None. If its Python type forbids it from being
        mangled, throw a ``CaptureError``.
        """
        ...

class TrivialCapturer(Capturer):
    """A ``Capturer`` that never captures any Python variables."""

    def shadows_python_variable(self, var_name: str) -> bool:
        return False

    def capture(self, var_name: str) -> Optional[str]:
        return None

def trivial_name_generator():
    """Return a ``name_generator`` that leaves names unchanged."""
    return lambda name: name

def convert_func_ast(ast_kind: AstKind, module: ast.Module,
                     name_generator: Callable[[str], str], capturer: Capturer,
                     filename: str = '', line_offset: int = 0,
                     col_offset: int = 0) \
                    -> QpuFunctionDef | ClassicalFunctionDef:
    """
    Take in a Python AST for a function parsed with ``ast.parse(mode='exec')``
    and return a ``Kernel`` Qwerty AST node.

    The ``line_offset`` and `col_offset`` are useful (respectively) because a
    ``@qpu``/``@classical`` kernel may begin after the first line of the file,
    and the caller may de-indent source code to avoid angering ``ast.parse()``.
    """
    if ast_kind == AstKind.QPU:
        return convert_qpu_func_ast(module, name_generator, capturer, filename,
                                    line_offset, col_offset)
    elif ast_kind == AstKind.CLASSICAL:
        return convert_classical_func_ast(module, name_generator, capturer,
                                          filename, line_offset, col_offset)
    else:
        raise ValueError('unknown AST type {}'.format(ast_kind))

REG_TYPES = {'bit': RegKind.Bit,
             'qubit': RegKind.Qubit}

class BaseVisitor:
    """
    Common Python AST visitor for both ``@classical`` and ``@qpu`` kernels.
    """

    def __init__(self,
                 expr_class: type[QpuExpr | ClassicalExpr],
                 stmt_class: type[QpuStmt | ClassicalStmt],
                 func_class: type[QpuFunctionDef | ClassicalFunctionDef],
                 name_generator: Callable[[str], str],
                 capturer: Capturer,
                 filename: str = '', line_offset: int = 0,
                 col_offset: int = 0):
        """
        Constructor. The ``name_generator`` argument mangles the Python AST
        name to produce a Qwerty AST name.
        """
        self._expr_class = expr_class
        self._stmt_class = stmt_class
        self._func_class = func_class
        # Used when constructing DimVars, since each contains the function name
        self._func_name = None
        self._next_internal_dim_var_id = 0
        self._internal_dim_vars = []
        # We push to this when we parsing the body of a Repeat or a
        # BasisAliasRecDef and pop when we're done
        self._macro_dimvar_stack = []
        self.name_generator = name_generator
        self.capturer = capturer
        self.filename = filename
        self.line_offset = line_offset
        self.col_offset = col_offset
        self.frame = get_frame()

    def get_node_row_col(self, node: ast.AST):
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            row = node.lineno + self.line_offset
            col = node.col_offset + 1 + self.col_offset
            return row, col
        else:
            return None, None

    def get_debug_loc(self, node: ast.AST) -> DebugLoc:
        """
        Extract line and column number from a Python AST node and return a
        Qwerty DebugInfo instance.
        """
        row, col = self.get_node_row_col(node)
        dbg = DebugLoc(self.filename, row or 0, col or 0)
        set_dbg_frame(dbg, self.frame)
        return dbg

    def allocate_internal_dim_var(self, dbg: Optional[DebugLoc]) -> DimExpr:
        """
        If we are inside a function, allocate a temporary dimension
        variable. If not, crash.
        """
        assert self._func_name is not None, 'cannot allocate internal ' \
                                            'dimvar outside function'
        dim_var_name = f'__{self._next_internal_dim_var_id}'
        self._internal_dim_vars.append(dim_var_name)
        self._next_internal_dim_var_id += 1
        dim_var = DimVar.new_func_var(dim_var_name, self._func_name)
        return DimExpr.new_var(dim_var, dbg)

    def extract_dimvar_expr(self, node: ast.AST) -> DimExpr:
        """
        Return a dimension variable expression given a subtree of the AST, for
        example::

            std**(2*N + 1)
                  ^^^^^^^
        """

        dbg = self.get_debug_loc(node)

        if isinstance(node, ast.Name):
            name = node.id
            captured_dim_expr = self.capture_dim_expr(name, dbg)

            if captured_dim_expr is not None:
                return captured_dim_expr
            elif name in self._macro_dimvar_stack:
                var = DimVar.new_macro_param(name)
            elif self._func_name is not None:
                var = DimVar.new_func_var(name, self._func_name)
            else:
                raise QwertySyntaxError('Cannot use dimension variables '
                                        'outside functions or macro '
                                        f'definitions, but found {name}', dbg)

            return DimExpr.new_var(var, dbg)
        elif isinstance(node, ast.Constant) \
                and isinstance(node.value, int):
            return DimExpr.new_const(node.value, dbg)
        elif isinstance(node, ast.BinOp) \
                and isinstance(node.op, ast.Add):
            left = self.extract_dimvar_expr(node.left)
            right = self.extract_dimvar_expr(node.right)
            return DimExpr.new_sum(left, right, dbg)
        elif isinstance(node, ast.BinOp) \
                and isinstance(node.op, ast.Sub):
            left = self.extract_dimvar_expr(node.left)
            right = self.extract_dimvar_expr(node.right)
            right_neg = DimExpr.new_neg(right, dbg)
            return DimExpr.new_sum(left, right_neg, dbg)
        elif isinstance(node, ast.BinOp) \
                and isinstance(node.op, ast.Mult):
            left = self.extract_dimvar_expr(node.left)
            right = self.extract_dimvar_expr(node.right)
            return DimExpr.new_prod(left, right, dbg)
        elif isinstance(node, ast.BinOp) \
                and isinstance(node.op, ast.Pow):
            base = self.extract_dimvar_expr(node.left)
            pow_ = self.extract_dimvar_expr(node.right)
            return DimExpr.new_pow(base, pow_, dbg)
        elif isinstance(node, ast.UnaryOp) \
                and isinstance(node.op, ast.USub):
            val = self.extract_dimvar_expr(node.operand)
            return DimExpr.new_neg(val, dbg)
        else:
            raise QwertySyntaxError('Unsupported dimension variable '
                                    'expression', dbg)

    #def extract_comma_sep_dimvar_expr(self, node: ast.AST) -> List[DimVarExpr]:
    #    if isinstance(node, ast.Tuple):
    #        tuple_ = node
    #        return [self.extract_dimvar_expr(elt) for elt in tuple_.elts]
    #    else:
    #        return [self.extract_dimvar_expr(node)]

    def extract_type_literal(self, node: ast.AST) -> Type:
        """
        Parse a type annotation and return a Qwerty AST Type.
        """
        if isinstance(node, ast.Name):
            type_name = node.id
            one = DimExpr.new_const(1, self.get_debug_loc(node))
            if type_name in REG_TYPES:
                reg_kind = REG_TYPES[type_name]
                return Type.new_reg(reg_kind, one)
            elif type_name == 'qfunc':
                return Type.new_func(Type.new_reg(RegKind.Qubit, one),
                                     Type.new_reg(RegKind.Qubit, one))
            elif type_name == 'rev_qfunc':
                return Type.new_rev_func(Type.new_reg(RegKind.Qubit, one))
            else:
                raise QwertySyntaxError('Unknown type name {} found'
                                        .format(type_name),
                                        self.get_debug_loc(node))
        elif isinstance(node, ast.Subscript) \
                and isinstance(node.value, ast.Name) \
                and (type_name := node.value.id) in ('bit', 'qubit', 'qfunc', 'rev_qfunc'):
            arg = node.slice
            if type_name in REG_TYPES:
                dim = self.extract_dimvar_expr(arg)
                reg_kind = REG_TYPES[type_name]
                return Type.new_reg(reg_kind, dim)
            elif type_name == 'rev_qfunc':
                dim = self.extract_dimvar_expr(arg)
                return Type.new_rev_func(Type.new_reg(RegKind.Qubit, dim),
                                         Type.new_reg(RegKind.Qubit, dim))
            elif type_name == 'qfunc':
                if isinstance((dims_node := arg), ast.Tuple):
                    if len(dims_node_elts := dims_node.elts) != 2:
                        raise QwertySyntaxError('Wrong number of params to '
                                                'qfunc. Expected 2 but got '
                                                f'{dims_node_elts}',
                                                self.get_debug_loc(dims_node))

                    in_dim_node, out_dim_node = dims_node_elts
                    in_dim = self.extract_dimvar_expr(in_dim_node)
                    out_dim = self.extract_dimvar_expr(out_dim_node)
                    return Type.new_func(Type.new_reg(RegKind.Qubit, in_dim),
                                         Type.new_reg(RegKind.Qubit, out_dim))
                else:
                    dim = self.extract_dimvar_expr(arg)
                    return Type.new_func(Type.new_reg(RegKind.Qubit, dim),
                                         Type.new_reg(RegKind.Qubit, dim))
            else:
                raise QwertySyntaxError('Unknown type name {}[...] found'
                                        .format(type_name),
                                        self.get_debug_loc(node))
        else:
            raise QwertySyntaxError('Unknown type',
                                    self.get_debug_loc(node))

    def is_bit_literal(self, call: ast.Call) -> bool:
        return (isinstance(subscript := call.func, ast.Subscript)
                and isinstance(name := subscript.value, ast.Name)
                and name.id == 'bit')

    def extract_bit_literal(self, call: ast.Call) -> QpuExpr | ClassicalExpr:
        subscript = call.func
        name = subscript.value
        dbg = self.get_debug_loc(call)

        if not isinstance(dim_const := subscript.slice, ast.Constant) \
                or not isinstance(dim := dim_const.value, int):
            dim_dbg = self.get_debug_loc(dim_const)
            raise QwertySyntaxError('Dimension N to a bit literal '
                                    '`bit[N](0b1101)` must be an integer '
                                    'constant.', dim_dbg)
        if len(call.args) != 1 \
                or not isinstance(bits_const := call.args[0], ast.Constant) \
                or not isinstance(bits := bits_const.value, int):
            # Try to choose a useful source code location to point them to
            if not call.args:
                bits_dbg = dbg
            elif len(call.args) > 1:
                bits_dbg = self.get_debug_loc(call.args[1])
            else:
                bits_dbg = self.get_debug_loc(bits_const)

            raise QwertySyntaxError('A bit literal `bit[N](0bxxxx)` '
                                    'requires only constant bits `0bxxxx` '
                                    'between the parentheses.', bits_dbg)

        return self._expr_class.new_bit_literal(bits, dim, dbg)

    def visit_macro(self,
                    macro_pat: ast.AST,
                    macro_name: str,
                    macro_rhs: ast.AST,
                    dbg: DebugLoc) -> QpuStmt | ClassicalStmt:
        """
        Visit a macro definition, as in::

            b.measure = __MEASURE__(b)

        Here, ``b`` (on the left) is ``macro_pat``, then ``measure`` is
        ``macro_name``, and ``macro_rhs`` is ``__MEASURE__(b)``.

        This default implementation throws an error, but subclasses should
        override this if they support macros.
        """
        return QwertySyntaxError('Macros are not supported here.', dbg)

    def visit_parametric_assign(self,
                                lhs_name: str,
                                lhs_param: ast.AST,
                                rhs: ast.AST,
                                dbg: DebugLoc) -> QpuStmt | ClassicalStmt:
        """
        Visit a parametric assignment statement, as in::

            fourier[N] = fourier[N-1] // std.revolve

        Here, ``fourier`` (on the left) is ``lhs_name``, then ``N`` (on the
        left) is ``lhs_param``, and ``rhs`` is ``fourier[N-1] // std.revolve``.

        This default implementation throws an error, but subclasses should
        override this if they support this syntax.
        """
        return QwertySyntaxError('This assignment syntax is not supported '
                                 'here.', dbg)

    def visit_Module(self, module: ast.Module):
        """
        Root node of Python AST
        """
        # No idea what this is, so conservatively reject it
        if module.type_ignores:
            raise QwertySyntaxError('I do not understand type_ignores, but '
                                    'they were specified in a Python module',
                                    self.get_debug_loc(module))

        if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
            raise QwertySyntaxError('Expected exactly 1 FunctionDef in '
                                    'module body',
                                    self.get_debug_loc(module))

        func_def = module.body[0]
        return self.visit_FunctionDef(func_def)

    def visit_Expr(self, expr_stmt: ast.Expr):
        """
        Statement that consists only of an expression
        """
        expr = self.visit(expr_stmt.value)
        return self._stmt_class.new_expr(expr)

    def base_visit_FunctionDef(self, func_def: ast.FunctionDef,
                               decorator_name: str) \
                              -> QpuFunctionDef | ClassicalFunctionDef:
        """
        Common code for processing the function Python AST node for both
        ``@classical`` and ``@qpu`` kernels.
        """
        func_name = func_def.name
        generated_func_name = self.name_generator(func_name)
        # Used when constructing DimVars, since each contains the function name
        self._func_name = generated_func_name

        # Sanity check for unsupported features
        for arg_prop in ('posonlyargs', 'vararg', 'kwonlyargs', 'kw_defaults',
                         'kwarg', 'defaults'):
            if getattr(func_def.args, arg_prop):
                raise QwertySyntaxError('{}() uses unsupported argument feature {}'
                                        .format(func_name, arg_prop),
                                        self.get_debug_loc(func_def))

        is_rev_dec = lambda dec: isinstance(dec, ast.Name) and dec.id == 'reversible'
        if (n_decorators := len(func_def.decorator_list)) > 2:
            raise QwertySyntaxError('Wrong number of decorators ({} > 2) '
                                    'for {}()'.format(n_decorators,
                                                      func_name),
                                    self.get_debug_loc(func_def))
        elif n_decorators == 2:
            rev_decorator, our_decorator = func_def.decorator_list
            if not is_rev_dec(rev_decorator):
                # swap
                our_decorator, rev_decorator = rev_decorator, our_decorator
            if not is_rev_dec(rev_decorator):
                raise QwertySyntaxError('Unknown decorator {} on {}()'
                                        .format(rev_decorator,
                                                func_name),
                                        self.get_debug_loc(func_def))
            # By this point, one of the decorators is @reversible
            is_rev = True
        elif n_decorators == 1:
            our_decorator, = func_def.decorator_list
            is_rev = False
        else: # n_decorators == 0
            raise QwertySyntaxError('No decorators (e.g., @{}) for {}()'
                                    .format(decorator_name, func_name),
                                    self.get_debug_loc(func_def))

        # First, unwrap the outer call with @qpu(prelude=my_prelude)
        #                                       ^^^^^^^^^^^^^^^^^^^^
        if isinstance(our_decorator, ast.Call) \
                and not our_decorator.args \
                and len(our_decorator.keywords) == 1 \
                and our_decorator.keywords[0].arg == 'prelude':
            # We already got the value of the prelude kwarg earlier so we can
            # ignore it here.
            our_decorator = our_decorator.func

        # Then try to determine this function's type variables
        is_ours = lambda node: isinstance(node, ast.Name) \
                               and node.id == decorator_name
        is_free_tv = lambda node: isinstance(node, ast.Name)
        is_explicit_tv = lambda node: \
            isinstance(call := node, ast.Call) \
                         and len(call.args) == 1 and not call.keywords \
                         and isinstance(call.func, ast.Name)
        is_tv = lambda node: is_free_tv(node) or is_explicit_tv(node)
        get_tv = lambda node: node.id if is_free_tv(node) \
                              else node.func.id

        if is_ours(our_decorator):
            dim_vars = []
            tvs_has_explicit_value = []
        elif isinstance(our_decorator, ast.Subscript) and is_ours(our_decorator.value) \
                and isinstance(our_decorator.slice, ast.List) \
                and all(is_tv(e) for e in our_decorator.slice.elts):
            tvs = our_decorator.slice.elts
            dim_vars = [get_tv(name) for name in tvs]
            tvs_has_explicit_value = [is_explicit_tv(dv) for dv in tvs]
        else:
            raise QwertySyntaxError('Mysterious non-@{}[[N,M]] decorator for '
                                    '{}()'.format(decorator_name,
                                                  func_name),
                                    self.get_debug_loc(our_decorator))

        # TODO: do soemthing with dim_vars and tvs_has_explicit_value

        # Next, figure out argument types
        args = []

        for arg in func_def.args.args:
            # .args.args.arg.arg... ARGH!!!!
            arg_name = arg.arg
            arg_type = arg.annotation

            if arg.type_comment:
                # What even is this thing? Conservatively rejecting it
                raise QwertySyntaxError('Unsupported type comment found on {} '
                                        'argument of {}()'
                                        .format(arg_name, func_name),
                                        self.get_debug_loc(arg))
            if arg_type:
                actual_arg_type = self.extract_type_literal(arg_type)
            else:
                actual_arg_type = None

            args.append((actual_arg_type, arg_name))

        # Now, figure out return type
        if func_def.returns:
            ret_type = self.extract_type_literal(func_def.returns)
        else:
            ret_type = None

        # Great, now we have everything we need to build the AST node...
        dbg = self.get_debug_loc(func_def)

        # ...except traversing the function body
        body = self.visit(func_def.body)

        # We may have allocated internal dimvars while walking through body, so
        # include them in the AST
        dim_vars += self._internal_dim_vars

        return self._func_class(generated_func_name, args, ret_type, body, is_rev, dim_vars, dbg)

    def visit_list(self, nodes: List[ast.AST]):
        """
        Convenience function to visit each node in a ``list`` and return the
        results of each as a new list.
        """
        return [self.visit(node) for node in nodes]

    def visit_Return(self, ret: ast.Return):
        """
        Convert a Python ``return`` statement into a Qwerty AST ``Return``
        node.
        """
        dbg = self.get_debug_loc(ret)
        expr = self.visit(ret.value)
        return self._stmt_class.new_return(expr, dbg)

    def visit_plain_assign(self, lhs_name: str, rhs: ast.AST, dbg: DebugLoc) \
                          -> QpuStmt | ClassicalStmt:
        """
        Convert a boring old assignment (``lhs_name = rhs``) to a Qwerty
        statement. This is a separate method so that it can be overriden.
        """
        expr = self.visit(rhs)

        if self.capturer.shadows_python_variable(lhs_name):
            raise QwertySyntaxError('Cannot define a variable '
                                    f'({lhs_name}) that shadows a Python '
                                    'variable.', dbg)

        return self._stmt_class.new_assign(lhs_name, expr, dbg)

    def visit_Assign(self, assign: ast.Assign) -> QpuStmt | ClassicalStmt:
        """
        Convert a Python assignment statement::

            q = '0'

        into an ``Assign`` Qwerty AST node.

        A destructuring assignment::

            q1, q2 = '01'

        is converted into a ``UnpackAssign`` Qwerty AST node.
        """
        dbg = self.get_debug_loc(assign)
        if len(assign.targets) != 1:
            # Something like a = b = c = '0'
            raise QwertySyntaxError("Assigning to multiple targets (e.g., "
                                    "`a = b = '0'`) are not supported. Please "
                                    "write `a = '0'` and then `b = '0'` "
                                    "instead",
                                    dbg)
        tgt, = assign.targets

        if isinstance(tgt, ast.Name):
            lhs_name = tgt.id
            rhs = assign.value
            return self.visit_plain_assign(lhs_name, rhs, dbg)
        elif isinstance(tgt, ast.Tuple) \
                and all(isinstance(elt, ast.Name) for elt in tgt.elts):
            var_names = [name.id for name in tgt.elts]

            if len(var_names) < 2:
                raise QwertySyntaxError('Unpacking assignment must have '
                                        'at least two names on the left-hand '
                                        'side', dbg)

            for var_name in var_names:
                if self.capturer.shadows_python_variable(var_name):
                    raise QwertySyntaxError('Cannot unpack into a variable '
                                            f'({var_name}) that shadows a '
                                            'Python variable.', dbg)

            expr = self.visit(assign.value)
            return self._stmt_class.new_unpack_assign(var_names, expr, dbg)
        elif isinstance(attr := tgt, ast.Attribute):
            macro_pat = attr.value
            macro_name = attr.attr
            macro_rhs = assign.value
            return self.visit_macro(macro_pat, macro_name, macro_rhs, dbg)
        elif isinstance(subscript := tgt, ast.Subscript) \
                and isinstance(lhs_name_node := subscript.value, ast.Name):
            lhs_name = lhs_name_node.id
            lhs_param = subscript.slice
            rhs = assign.value
            return self.visit_parametric_assign(lhs_name, lhs_param, rhs, dbg)
        else:
            raise QwertySyntaxError('Unknown assignment syntax', dbg)

    #def visit_AnnAssign(self, assign: ast.AnnAssign):
    #    """
    #    Throw an error for the Python type-annotated assignment statement,
    #    since it is unnecessary::

    #        q: qubit = '0'
    #    """
    #    dbg = self.get_debug_loc(assign)
    #    raise QwertySyntaxError('Typed assignments, i.e.\n'
    #                            '\tsimple -- q: qubit = v\n'
    #                            '\tdestructuring -- (a, b): (qubit, qubit[2]) '
    #                            '= v\n'
    #                            '\tsubscript typing -- a[1]: bit = x\n'
    #                            '\tor even just wrapping the name in parens '
    #                            '-- (a): bit = y\n'
    #                            'are currently unsupported. Please give '
    #                            'variables simple names without annotations '
    #                            'for now.',
    #                            dbg)

    #def visit_AugAssign(self, assign: ast.AugAssign):
    #    """
    #    Throw an error for the Python augmented assignment statement, since it
    #    is currently not supported::

    #        q += '0'
    #        q *= '0'
    #        q -= '0'
    #        # ...etc.
    #    """
    #    raise QwertySyntaxError('Qwerty does not support mutating values', dbg)

    def capture(self, var_name: str, dbg: DebugLoc) \
               -> Optional[CapturedValue]:
        try:
            captured = self.capturer.capture(var_name)
        except CaptureError as err:
            var_type_name = err.type_name
            raise QwertySyntaxError(f'The Python object named {var_name} is '
                                    'referenced, but it has Python type '
                                    f'{var_type_name}, which cannot be used '
                                    'in Qwerty kernels.', dbg)

        return captured

    def capture_symbol_name(self, var_name: str, dbg: DebugLoc) \
                           -> Optional[str]:
        captured = self.capture(var_name, dbg)

        if captured is not None:
            return captured.to_symbol_name(dbg)
        else:
            return None

    def capture_expr(self, var_name: str, dbg: DebugLoc) \
                    -> Optional[QpuExpr | ClassicalExpr]:
        captured = self.capture(var_name, dbg)

        if captured is not None:
            return captured.to_expr(self._expr_class, dbg)
        else:
            return None

    def capture_dim_expr(self, var_name: str, dbg: DebugLoc) \
                        -> Optional[DimExpr]:
        captured = self.capture(var_name, dbg)

        if captured is not None:
            return captured.to_dim_expr(dbg)
        else:
            return None

    def capture_float_expr(self, var_name: str, dbg: DebugLoc) \
                          -> Optional[FloatExpr]:
        captured = self.capture(var_name, dbg)

        if captured is not None:
            return captured.to_float_expr(dbg)
        else:
            return None

    def visit_Name(self, name: ast.Name) -> QpuExpr | ClassicalExpr:
        """
        Convert a Python AST identitifer node into a Qwerty variable name AST
        node. For example, ``foobar`` becomes a Qwerty ``Variable`` AST node.
        """
        var_name = name.id
        dbg = self.get_debug_loc(name)
        captured_expr = self.capture_expr(var_name, dbg)

        if captured_expr is not None:
            return captured_expr
        else:
            return self._expr_class.new_variable(var_name, dbg)

    def base_visit(self, node: ast.AST):
        """
        Convert a Python AST node into a Qwerty AST Node (and return the
        latter).
        """
        if isinstance(node, list):
            return self.visit_list(node)
        elif isinstance(node, ast.Expr):
            return self.visit_Expr(node)
        elif isinstance(node, ast.Return):
            return self.visit_Return(node)
        elif isinstance(node, ast.Assign):
            return self.visit_Assign(node)
        elif isinstance(node, ast.Name):
            return self.visit_Name(node)
        #elif isinstance(node, ast.AnnAssign):
        #    return self.visit_AnnAssign(node)
        #elif isinstance(node, ast.AugAssign):
        #    return self.visit_AugAssign(node)
        # Commenting these for now, since we can't handle nested functions, and
        # a nested module doesn't make much sense
        #elif isinstance(node, ast.Module):
        #    return self.visit_Module(node)
        #elif isinstance(node, ast.FunctionDef):
        #    return self.visit_FunctionDef(node)
        else:
            node_name = type(node).__name__
            raise QwertySyntaxError(f'Unknown Python AST node {node_name}',
                                    self.get_debug_loc(node))

#################### @QPU DSL ####################

# NOTE: For now, users should be able to do either +- or pm
#STATE_CHAR_MAPPING = {
#    '+': (PLUS, X),
#    '-': (MINUS, X),
#    'p': (PLUS, X),
#    'm': (MINUS, X),
#    'i': (PLUS, Y),
#    'j': (MINUS, Y),
#    '0': (PLUS, Z),
#    '1': (MINUS, Z),
#}
#
#EMBEDDING_KINDS = (
#    EMBED_XOR,
#    EMBED_SIGN,
#    EMBED_INPLACE,
#)
#
#EMBEDDING_KEYWORDS = {embedding_kind_name(e): e
#                      for e in EMBEDDING_KINDS}
#
#RESERVED_KEYWORDS = {'id', 'discard', 'discardz', 'measure', 'flip'}

# Mapping of intrinsic names to the number of arguments required
INTRINSICS = {
    '__MEASURE__': 1,
    '__DISCARD__': 0,
    '__EMBED_XOR__': 1,
    '__EMBED_SIGN__': 1,
    '__EMBED_INPLACE__': 1,
}

EMBED_KINDS = {
    '__EMBED_XOR__': EmbedKind.Xor,
    '__EMBED_SIGN__': EmbedKind.Sign,
    '__EMBED_INPLACE__': EmbedKind.InPlace,
}

VECTOR_ATOM_INTRINSICS = {'__SYM_STD0__': Vector.new_zero_vector,
                          '__SYM_STD1__': Vector.new_one_vector,
                          '__SYM_PAD__': Vector.new_pad_vector,
                          '__SYM_TARGET__': Vector.new_target_vector}

class QpuVisitor(BaseVisitor):
    """
    Python AST visitor for syntax specific to the ``@qpu`` DSL.
    """

    def __init__(self, name_generator: Callable[[str], str],
                 capturer: Capturer, filename: str = '', line_offset: int = 0,
                 col_offset: int = 0):
        super().__init__(QpuExpr, QpuStmt, QpuFunctionDef, name_generator,
                         capturer, filename, line_offset, col_offset)

    def visit_plain_assign(self, lhs_name: str, rhs: ast.AST, dbg: DebugLoc) \
                          -> QpuStmt | ClassicalStmt:
        """
        Override `BaseVisitor::visit_plain_assign()` so that we can support
        basis aliases too.
        """
        try:
            # Ban singleton vectors to avoid interpreting ``zero = '0'`` as a
            # basis alias defintion. Also ban aliases to avoid ``q1 = q2`` from
            # being parsed as a basis alias definition. These restrictions are
            # pretty harmless because I expect basis alias definitions to be
            # sparsely used.
            rhs_basis = self.extract_basis(rhs, allow_singleton=False,
                                           allow_alias=False)
        # TODO: do a more granular catch here
        except QwertySyntaxError:
            return super().visit_plain_assign(lhs_name, rhs, dbg)
        else:
            return QpuStmt.new_basis_alias_def(lhs_name, rhs_basis, dbg)

    def visit_macro(self,
                    macro_pat: ast.AST,
                    macro_name: str,
                    macro_rhs: ast.AST,
                    dbg: DebugLoc) -> QpuStmt | ClassicalStmt:
        """
        Override of ``BaseVisitor::visit_macro()` that creates the appropriate
        metaQwerty AST nodes.
        """
        if macro_name == 'sym' \
                and isinstance(string_const := macro_pat, ast.Constant) \
                and isinstance(sym := string_const.value, str):
            string_const_dbg = self.get_debug_loc(string_const)
            if (sym_len := len(sym)) != 1:
                raise QwertySyntaxError('Vector symbols must have length 1, '
                                        f'but found {sym_len} instead.', dbg)
            vec = self.extract_basis_vector(macro_rhs)
            return QpuStmt.new_vector_symbol_def(sym, vec, dbg)
        elif isinstance(attr := macro_pat, ast.Attribute) \
                and attr.attr == 'expr':
            macro_pat = attr.value
            macro_pat_dbg = self.get_debug_loc(macro_pat)

            if isinstance(macro_pat, ast.Name):
                macro_pat_name = macro_pat.id
                lhs_pat = ExprMacroPattern.new_any_expr(macro_pat_name,
                                                        macro_pat_dbg)
            else:
                raise QwertySyntaxError('Unknown experssion pattern syntax',
                                        macro_pat_dbg)

            rhs = self.visit(macro_rhs)
            return QpuStmt.new_expr_macro_def(lhs_pat, macro_name, rhs, dbg)
        else:
            macro_pat_dbg = self.get_debug_loc(macro_pat)
            if isinstance(macro_pat, ast.Name):
                macro_pat_name = macro_pat.id
                lhs_pat = BasisMacroPattern.new_any_basis(macro_pat_name,
                                                          macro_pat_dbg)
            elif isinstance(set_ := macro_pat, ast.Set) \
                    and all(isinstance(elt, ast.Name) for elt in set_.elts):
                vec_names = [elt.id for elt in set_.elts]
                lhs_pat = BasisMacroPattern.new_basis_literal(vec_names,
                                                              macro_pat_dbg)
            else:
                raise QwertySyntaxError('Unknown basis pattern syntax',
                                        macro_pat_dbg)

            try:
                # ``allow_macro=False`` avoids a syntactic ambiguitity
                rhs_bgen = self.extract_basis_generator(macro_rhs,
                                                        allow_macro=False)
            # TODO: do a more granular catch here
            except QwertySyntaxError:
                rhs_expr = self.visit(macro_rhs)
                return QpuStmt.new_basis_macro_def(lhs_pat, macro_name,
                                                   rhs_expr, dbg)
            else:
                return QpuStmt.new_basis_generator_macro_def(lhs_pat,
                                                             macro_name,
                                                             rhs_bgen, dbg)

    def visit_parametric_assign(self,
                                lhs_name: str,
                                lhs_param: ast.AST,
                                rhs: ast.AST,
                                dbg: DebugLoc) -> QpuStmt | ClassicalStmt:
        """
        Override of ``BaseVisitor::visit_parametric_assign()`` that creates a
        ``BasisAliasRecDef`` metaQwerty AST node.
        """
        lhs_param_dbg = self.get_debug_loc(lhs_param)
        if isinstance(const := lhs_param, ast.Constant) \
                and isinstance(base_val := const.value, int):
            rec_name = None
            param = RecDefParam.new_base(base_val)
        elif isinstance(name := lhs_param, ast.Name):
            rec_name = name.id
            param = RecDefParam.new_rec(rec_name)
        else:
            raise QwertySyntaxError('Unknown syntax for parameter in '
                                    'recursive basis alias definition',
                                    lhs_param_dbg)

        if rec_name is not None:
            self._macro_dimvar_stack.append(rec_name)
        rhs_basis = self.extract_basis(rhs)
        if rec_name is not None:
            self._macro_dimvar_stack.pop()

        return QpuStmt.new_basis_alias_rec_def(lhs_name, param, rhs_basis, dbg)

    def extract_qubit_literal(self, node: ast.AST) -> QpuExpr:
        return QpuExpr.new_qlit(self.extract_basis_vector(node))

    def is_vector_atom_intrinsic(self, node: ast.AST) -> bool:
        return isinstance(call := node, ast.Call) \
            and not call.args \
            and not call.keywords \
            and isinstance(name := call.func, ast.Name) \
            and (intrinsic_name := name.id) in VECTOR_ATOM_INTRINSICS

    def extract_vector_atom_intrinsic(self, call: ast.AST) -> Vector:
        dbg = self.get_debug_loc(call)
        name = call.func
        intrinsic_name = name.id
        return VECTOR_ATOM_INTRINSICS[intrinsic_name](dbg)

    def extract_basis_vector(self, node: ast.AST) -> Vector:
        """
        Convert a Python AST node to a Qwerty AST ``Vector`` node.
        """

        dbg = self.get_debug_loc(node)

        if isinstance(name := node, ast.Name):
            alias_name = name.id
            return Vector.new_vector_alias(alias_name, dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.extract_basis_vector(node.left)
            right = self.extract_basis_vector(node.right)
            return Vector.new_uniform_vector_superpos(left, right, dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            left = self.extract_basis_vector(node.left)
            right = self.extract_basis_vector(node.right)
            angle_deg = FloatExpr.new_const(180.0, dbg)
            right_neg = Vector.new_vector_tilt(right, angle_deg, dbg)
            return Vector.new_uniform_vector_superpos(left, right_neg, dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left = self.extract_basis_vector(node.left)
            right = self.extract_basis_vector(node.right)
            return Vector.new_vector_bi_tensor(left, right, dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            left = self.extract_basis_vector(node.left)
            right = self.extract_dimvar_expr(node.right)
            return Vector.new_vector_broadcast_tensor(left, right, dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            q = self.extract_basis_vector(node.left)
            angle_deg = self.extract_float_expr(node.right)
            return Vector.new_vector_tilt(q, angle_deg, dbg)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            q = self.extract_basis_vector(node.operand)
            angle_deg = FloatExpr.new_const(180.0, dbg)
            return Vector.new_vector_tilt(q, angle_deg, dbg)
        elif self.is_vector_atom_intrinsic(node):
            return self.extract_vector_atom_intrinsic(node)
        elif isinstance(node, ast.Constant) and node.value == '':
            return Vector.new_vector_unit(dbg)
        elif isinstance(node, ast.Constant):
            return functools.reduce(
                lambda acc, atom: Vector.new_vector_bi_tensor(acc, atom, dbg),
                (Vector.new_vector_symbol(sym, dbg) for sym in node.value))
        else:
            node_name = type(node).__name__
            raise QwertySyntaxError('Unknown basis vector or qubit literal syntax {}'
                                    .format(node_name), dbg)

    def extract_basis_generator(self, node: ast.AST, *,
                                allow_macro=True) -> BasisGenerator:
        """
        Convert a Python AST node into a ``BasisGenerator`` Qwerty AST node.
        Passing ``allow_macro=False`` bans use of basis generator macros, which
        is useful for avoiding syntactic ambiguities for basis macro
        definitions versus basis generator macro definitions.
        """
        dbg = self.get_debug_loc(node)

        if isinstance(call := node, ast.Call) \
                and isinstance(func_name := call.func, ast.Name) \
                and func_name.id == '__REVOLVE__':
            n_args = len(call.args)
            if n_args != 2:
                raise QwertySyntaxError(f'Wrong number of arguments {n_args} '
                                        '!= 2 to __REVOLVE__ intrinsic', dbg)
            arg1, arg2 = call.args
            v1 = self.extract_basis_vector(arg1)
            v2 = self.extract_basis_vector(arg2)
            return BasisGenerator.new_revolve(v1, v2, dbg)
        elif allow_macro and isinstance(attr := node, ast.Attribute):
            name = attr.attr
            arg = self.extract_basis(attr.value)
            return BasisGenerator.new_basis_generator_macro(name, arg, dbg)
        else:
            node_name = type(node).__name__
            raise QwertySyntaxError('Unknown basis generator syntax {}'
                                    .format(node_name), dbg)

    def extract_basis(self, node: ast.AST, *, allow_singleton=True,
                      allow_alias=True) -> Basis:
        """
        Extract a Basis AST node from a Python AST node. The caller can apply
        some restrictions: first, setting ``allow_singleton=False`` means that
        singleton vectors such as ``'0'`` will not be interpreted as ``{'0'}``.
        The second available restriction is that passing ``allow_alias=False``
        will prevent identifiers such as ``pm`` from being interpreted as basis
        aliases; instead, you'd need to write ``{'p','m'}``. Both avoid
        syntactic ambiguities in different cases.
        """
        dbg = self.get_debug_loc(node)

        if allow_alias \
                and isinstance(identifier := node, ast.Name):
            alias_name = identifier.id
            return Basis.new_basis_alias(alias_name, dbg)
        elif isinstance(subscript := node, ast.Subscript) \
                and isinstance(name_node := subscript.value, ast.Name):
            name = name_node.id
            param = self.extract_dimvar_expr(subscript.slice)
            return Basis.new_basis_alias_rec(name, param, dbg)
        elif isinstance(node, ast.Set) and not node.elts:
            return Basis.new_empty_basis_literal(dbg)
        elif isinstance(node, ast.Set) and node.elts:
            vecs = [self.extract_basis_vector(elt) for elt in node.elts]
            return Basis.new_basis_literal(vecs, dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left = self.extract_basis(node.left,
                                      allow_singleton=allow_singleton,
                                      allow_alias=allow_alias)
            right = self.extract_basis(node.right,
                                       allow_singleton=allow_singleton,
                                       allow_alias=allow_alias)
            return Basis.new_basis_bi_tensor(left, right, dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
            basis = self.extract_basis(node.left,
                                       allow_singleton=allow_singleton,
                                       allow_alias=allow_alias)
            gen = self.extract_basis_generator(node.right)
            return Basis.new_apply_basis_generator(basis, gen, dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            basis = self.extract_basis(node.left,
                                       allow_singleton=allow_singleton,
                                       allow_alias=allow_alias)
            factor = self.extract_dimvar_expr(node.right)
            return Basis.new_basis_broadcast_tensor(basis, factor, dbg)
        elif allow_singleton and \
                ((isinstance(node, ast.UnaryOp)
                    and isinstance(node.op, ast.USub)) \
                or (isinstance(node, ast.BinOp)
                    and isinstance(node.op, ast.MatMult)) \
                or (isinstance(node, ast.Constant) \
                    and isinstance(node.value, str)) \
                or self.is_vector_atom_intrinsic(node)):
            return Basis.new_basis_literal([self.extract_basis_vector(node)], dbg)
        else:
            node_name = type(node).__name__
            raise QwertySyntaxError('Unknown basis syntax {}'
                                    .format(node_name), dbg)

    def extract_float_expr(self, node: ast.AST) -> FloatExpr:
        """
        Extract a float expression, like a tilt, for example::

            '1' @ (360/2)
                   ^^^^^^
        """
        dbg = self.get_debug_loc(node)

        if isinstance(name := node, ast.Name):
            var_name = name.id
            captured = self.capture_float_expr(var_name, dbg)
            if captured is not None:
                return captured
            else:
                dim_expr = self.extract_dimvar_expr(node)
                return FloatExpr.new_dim_expr(dim_expr, dbg)
        elif isinstance(node, ast.Constant) \
                and type(node.value) in (int, float):
            return FloatExpr.new_const(float(node.value), dbg)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = self.extract_float_expr(node.operand)
            return FloatExpr.new_neg(val, dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return FloatExpr.new_div(self.extract_float_expr(node.left),
                                     self.extract_float_expr(node.right),
                                     dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return FloatExpr.new_prod(self.extract_float_expr(node.left),
                                      self.extract_float_expr(node.right),
                                      dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            dim_expr = self.extract_dimvar_expr(node)
            return FloatExpr.new_dim_expr(dim_expr, dbg)
        #elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        #    return FloatBinaryOp(dbg, FLOAT_MOD,
        #                         self.extract_float_expr(node.left),
        #                         self.extract_float_expr(node.right))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return FloatExpr.new_sum(self.extract_float_expr(node.left),
                                     self.extract_float_expr(node.right),
                                     dbg)
        #elif isinstance(node, ast.Subscript) \
        #        and isinstance(node.value, ast.Name) \
        #        and isinstance(node.slice, ast.List) \
        #        and len(node.slice.elts) == 1:
        #    val = self.visit(node.value)
        #    idx = node.slice.elts[0]
        #    lower = self.extract_dimvar_expr(idx)
        #    upper = lower.copy()
        #    upper += DimVarExpr('', 1)
        #    return Slice(dbg, val, lower, upper)
        else:
            node_name = type(node).__name__
            raise QwertySyntaxError('Unsupported float expression {}'
                                    .format(node_name),
                                    self.get_debug_loc(node))

    def visit_FunctionDef(self, func_def: ast.FunctionDef) -> QpuFunctionDef:
        """
        Convert a ``@qpu`` kernel into a ``QpuKernel`` Qwerty AST node.
        """
        return super().base_visit_FunctionDef(func_def, 'qpu')

    def visit_Constant(self, const: ast.Constant):
        value = const.value
        if isinstance(value, str):
            # A top-level string must be a qubit literal
            return self.extract_qubit_literal(const)
        else:
            raise QwertySyntaxError('Unknown constant syntax',
                                    self.get_debug_loc(const))

    def visit_Subscript(self, subscript: ast.Subscript):
        """
        Convert a Python expression ``func[[N]`` into a Qwerty ``Instantiate``
        AST node.
        """
        dbg = self.get_debug_loc(subscript)
        value, slice_ = subscript.value, subscript.slice

        if isinstance(name_node := value, ast.Name) \
                and isinstance(list_ := slice_, ast.List) and list_.elts:
            n_elts = len(list_.elts)
            if n_elts != 1:
                raise QwertySyntaxError(
                    'Currently only one dimensional expression can be passed '
                    f'to instantiation, not {n_elts} expressions', dbg)

            name = name_node.id
            name_dbg = self.get_debug_loc(name_node)
            mangled_name = self.capture_symbol_name(name, name_dbg)
            if mangled_name is None:
                mangled_name = name

            param_node, = list_.elts
            param = self.extract_dimvar_expr(param_node)
            return QpuExpr.new_instantiate(mangled_name, param, dbg)
        else:
            raise QwertySyntaxError('Invalid instantiation syntax', dbg)

    def visit_BinOp(self, binOp: ast.BinOp):
        if isinstance(binOp.op, ast.Add):
            return self.visit_BinOp_Add(binOp)
        if isinstance(binOp.op, ast.Sub):
            return self.visit_BinOp_Sub(binOp)
        elif isinstance(binOp.op, ast.BitOr):
            return self.visit_BinOp_BitOr(binOp)
        elif isinstance(binOp.op, ast.BitXor):
            return self.visit_BinOp_BitXor(binOp)
        elif isinstance(binOp.op, ast.Mult):
            return self.visit_BinOp_Mult(binOp)
        elif isinstance(binOp.op, ast.RShift):
            return self.visit_BinOp_RShift(binOp)
        elif isinstance(binOp.op, ast.Pow):
            return self.visit_BinOp_Pow(binOp)
        #elif isinstance(binOp.op, ast.BitAnd):
        #    return self.visit_BinOp_BitAnd(binOp)
        #elif isinstance(binOp.op, ast.MatMult):
        #    return self.visit_BinOp_MatMult(binOp)
        else:
            op_name = type(binOp.op).__name__
            raise QwertySyntaxError('Unknown binary operation {}'
                                    .format(op_name),
                                    self.get_debug_loc(binOp))

    def try_extract_weighted_superpos(self, left: ast.AST, right: ast.AST) \
                                     -> Optional[list[tuple[float, Vector]]]:
        if isinstance(left_binop := left, ast.BinOp) \
                and isinstance(left_binop.op, ast.Mult) \
                and isinstance(right_binop := right, ast.BinOp) \
                and isinstance(right_binop.op, ast.Mult) \
                and isinstance(left_left := left_binop.left, ast.Constant) \
                and isinstance(left_prob := left_left.value, float) \
                and isinstance(right_left := right_binop.left, ast.Constant) \
                and isinstance(right_prob := right_left.value, float):
            left_vec_node = left.right
            right_vec_node = right.right
            left_vec = self.extract_basis_vector(left_vec_node)
            right_vec = self.extract_basis_vector(right_vec_node)
            elems = [(left_prob, left_vec), (right_prob, right_vec)]
            return elems
        else:
            return None

    def visit_BinOp_Add(self, binOp: ast.BinOp):
        # A top-level `+` expression must be a superpos. But we can do a check
        # here to see if it is a uniform superpos or a non-uniform one. For
        # now, we lower uniform superpositions to a qubit literal and
        # nonuniform ones to the dedicated ``NonUniformSuperpos`` node.
        dbg = self.get_debug_loc(binOp)
        left = binOp.left
        right = binOp.right
        if (elems := self.try_extract_weighted_superpos(left, right)) \
                is not None:
            return QpuExpr.new_non_uniform_superpos(elems, dbg)
        else:
            return self.extract_qubit_literal(binOp)

    def visit_BinOp_Sub(self, binOp: ast.BinOp):
        # A top-level `-` expression must be a qubit literal (a superpos with a
        # negative phase)
        return self.extract_qubit_literal(binOp)

    def visit_BinOp_BitOr(self, binOp: ast.BinOp):
        """
        Convert a Python bitwise OR expression into a Qwerty ``Pipe`` (function
        call) AST node. For example, ``t1 | t2`` becomes a ``Pipe`` node with
        two children.
        """
        left = self.visit(binOp.left)
        right = self.visit(binOp.right)
        dbg = self.get_debug_loc(binOp)
        return QpuExpr.new_pipe(left, right, dbg)

    def visit_BinOp_BitXor(self, binOp: ast.BinOp):
        """
        Convert a Python bitwise XOR expression into a Qwerty ``Ensemble`` AST
        node. For example, ``'0' ^ '1'`` becomes an ``Ensemble`` node with two
        elements, each with probability ``0.5``.
        """

        dbg = self.get_debug_loc(binOp)
        left = binOp.left
        right = binOp.right

        elems = self.try_extract_weighted_superpos(left, right)
        if elems is None:
            left_vec = self.extract_basis_vector(left)
            right_vec = self.extract_basis_vector(right)
            elems = [(0.5, left_vec), (0.5, right_vec)]

        return QpuExpr.new_ensemble(elems, dbg)

    def visit_BinOp_Mult(self, binOp: ast.BinOp):
        left = self.visit(binOp.left)
        right = self.visit(binOp.right)
        dbg = self.get_debug_loc(binOp)
        return QpuExpr.new_bi_tensor(left, right, dbg)

    #def visit_BinOp_BitAnd(self, binOp: ast.BinOp):
    #    """
    #    Convert a Python bitwise AND expression into a Qwerty ``Pred``
    #    (predication) AST node. For example, ``t1 & t2`` becomes a ``Pred``
    #    node with two children — one should be a basis and one should be a
    #    function.
    #    """
    #    basis = self.visit(binOp.left)
    #    body = self.visit(binOp.right)
    #    dbg = self.get_debug_loc(binOp)
    #    return Pred(dbg, basis, body)

    #def visit_BinOp_MatMult(self, binOp: ast.BinOp):
    #    """
    #    Convert a Python matrix multiplication expression into a Qwerty
    #    ``Phase`` (tilt) AST node. For example, ``t1 @ t2`` becomes a ``Phase``
    #    node with two children — the left should be a rev_qfunc[N] or qubit[N],
    #    and the right should be a float. If the right operand is in degrees
    #    (this is the default unless you write ``t1 @ rad(t2)``), then a
    #    conversion to radians is automatically synthesized.
    #    """
    #    if isinstance(call := binOp.right, ast.Call) \
    #            and isinstance(name := call.func, ast.Name) \
    #            and name.id in ('deg', 'rad'):
    #        unit = name.id
    #        if call.keywords:
    #            raise QwertySyntaxError(
    #                'Keyword arguments not supported for {}(...)'.format(unit),
    #                self.get_debug_loc(binOp))
    #        if len(call.args) != 1:
    #            raise QwertySyntaxError(
    #                'Wrong number of arguments {} != 1 passed to {}(...)'
    #                .format(len(call.args), unit),
    #                self.get_debug_loc(binOp))
    #        angle = call.args[0]
    #        # Set the debug location for the angle expression code to this
    #        # pseudo-function (deg() or rad())
    #        angle_conv_dbg_node = call
    #    else:
    #        unit = 'deg'
    #        angle = binOp.right
    #        angle_conv_dbg_node = binOp.right

    #    angle_expr = self.extract_float_expr(angle)
    #    if unit == 'deg':
    #        angle_conv_dbg = self.get_debug_loc(angle_conv_dbg_node)
    #        # Convert to radians: angle_expr/360 * 2*pi
    #        # The canonicalizer AST pass will fold this
    #        angle_expr = \
    #            FloatBinaryOp(
    #                angle_conv_dbg.copy(),
    #                FLOAT_MUL,
    #                FloatBinaryOp(
    #                    angle_conv_dbg.copy(),
    #                    FLOAT_DIV,
    #                    angle_expr,
    #                    FloatLiteral(
    #                        angle_conv_dbg.copy(), 360.0)),
    #                FloatLiteral(
    #                    angle_conv_dbg.copy(),
    #                    2*math.pi))

    #    dbg = self.get_debug_loc(binOp)
    #    lhs = self.visit(binOp.left)
    #    return Phase(dbg, angle_expr, lhs)

    def visit_BinOp_RShift(self, binOp: ast.BinOp):
        """
        Convert a Python right bit shift AST node to a Qwerty
        ``BasisTranslation`` AST node. For example, ``b1 >> b2`` becomes a
        ``BasisTranslation`` AST node with two basis children.
        """
        dbg = self.get_debug_loc(binOp)
        basis_in = self.extract_basis(binOp.left)
        basis_out = self.extract_basis(binOp.right)
        return QpuExpr.new_basis_translation(basis_in, basis_out, dbg)

    def visit_BinOp_Pow(self, binOp: ast.BinOp):
        """
        Convert a Python exponentiation AST node to a Qwerty
        ``BroadcastTensor`` AST node. For example, ``e**N`` becomes a
        ``BroadcastTensor`` node with a left expression child and a right
        ``DimExpr`` child.
        """
        dbg = self.get_debug_loc(binOp)
        val = self.visit(binOp.left)
        factor = self.extract_dimvar_expr(binOp.right)
        return QpuExpr.new_broadcast_tensor(val, factor, dbg)

    #def visit_UnaryOp(self, unaryOp: ast.UnaryOp):
    #    if isinstance(unaryOp.op, ast.USub):
    #        return self.visit_UnaryOp_USub(unaryOp)
    #    elif isinstance(unaryOp.op, ast.Invert):
    #        return self.visit_UnaryOp_Invert(unaryOp)
    #    else:
    #        op_name = type(unaryOp.op).__name__
    #        raise QwertySyntaxError('Unknown unary operation {}'
    #                                .format(op_name),
    #                                self.get_debug_loc(unaryOp))

    #def visit_UnaryOp_USub(self, unaryOp: ast.UnaryOp):
    #    """
    #    Convert a Python unary negation AST node into a Qwerty AST node tilting
    #    the operand by 180 degrees. For example, ``-f`` or ``-'0'``.
    #    """
    #    dbg = self.get_debug_loc(unaryOp)
    #    value = self.visit(unaryOp.operand)
    #    # Euler's identity, e^{iπ} = -1
    #    return Phase(dbg, FloatLiteral(dbg.copy(), math.pi), value)

    #def visit_UnaryOp_Invert(self, unaryOp: ast.UnaryOp):
    #    """
    #    Convert a Python unary bitwise complement AST node into a Qwerty
    #    ``Adjoint`` AST node. For example, ``~f`` becomes an ``Adjoint`` node
    #    with 1 child (``f``).
    #    """
    #    unary_operand = unaryOp.operand
    #    dbg = self.get_debug_loc(unaryOp)
    #    operand = self.visit(unary_operand)
    #    return Adjoint(dbg, operand)

    def visit_Set(self, set_: ast.Set):
        """
        Support syntactic sugar for basis translations. For example,
        ``{'0'>>'0'+'1'', '1'>>'0'-'1'}`` is syntactic sugar for
        ``{'0','1'} >> {'0'+'1', '0'-'1'}``. This method is not called for
        typical basis literals because they are handled by ``extract_basis()``.
        """
        dbg = self.get_debug_loc(set_)

        if not set_.elts:
            raise QwertySyntaxError("The list of replacements in the basis "
                                    "translation syntactic sugar `{'0'>>'0',"
                                    "'1'>>-'1'}` cannot be empty.", dbg)
        bvs_in = []
        bvs_out = []

        for elt in set_.elts:
            if isinstance(binop := elt, ast.BinOp) \
                    and isinstance(binop.op, ast.RShift):
                rshift = elt
                bvs_in.append(self.extract_basis_vector(rshift.left))
                bvs_out.append(self.extract_basis_vector(rshift.right))
            else:
                node_name = type(elt).__name__
                raise QwertySyntaxError("Invalid replacement in the "
                                        "elementwise basis translation "
                                        "syntactic sugar `{'0'>>'0',"
                                        "'1'>>-'1'}`. Expected a `>>` "
                                        f"but found a {node_name} instead.",
                                        dbg)

        basis_in = Basis.new_basis_literal(bvs_in, dbg)
        basis_out = Basis.new_basis_literal(bvs_out, dbg)
        return QpuExpr.new_basis_translation(basis_in, basis_out, dbg)

    def list_comp_helper(self, gen: Union[ast.GeneratorExp, ast.ListComp]):
        """
        Helper used by both ``visit_GeneratorExp()`` and ``visit_ListComp()``,
        since both Python AST nodes have near-identical fields.
        """
        dbg = self.get_debug_loc(gen)

        if len(gen.generators) != 1:
            raise QwertySyntaxError('Multiple generators are unsupported in '
                                    'Qwerty', dbg)
        comp, = gen.generators

        if comp.ifs:
            raise QwertySyntaxError('"if" not supported inside repeat '
                                    'construct', dbg)
        if comp.is_async:
            raise QwertySyntaxError('async generators not supported', dbg)

        if not isinstance(comp.target, ast.Name) \
                or not isinstance(comp.iter, ast.Call) \
                or not isinstance(comp.iter.func, ast.Name) \
                or comp.iter.func.id != 'range' \
                or len(comp.iter.args) != 1 \
                or comp.iter.keywords:
            raise QwertySyntaxError('Unsupported generator syntax (only '
                                    'basic "x for i in range(N)" is '
                                    'supported")', dbg)
        ub_node, = comp.iter.args

        iter_var = comp.target.id
        upper_bound = self.extract_dimvar_expr(ub_node)

        self._macro_dimvar_stack.append(iter_var)
        for_each = self.visit(gen.elt)
        self._macro_dimvar_stack.pop()

        return (for_each, iter_var, upper_bound, dbg)

    def visit_GeneratorExp(self, gen: ast.GeneratorExp):
        """
        Convert a Python generator expression AST node into a Qwerty ``Repeat``
        AST node. For example the highlighted part of the code::

            ... | (func for i in range(20)) | ...
                   ^^^^^^^^^^^^^^^^^^^^^^^

        is converted to a Repeat AST node. Here, ``range`` is a keyword whose
        operand is a dimension variable expression.
        """
        return QpuExpr.new_repeat(*self.list_comp_helper(gen))

    #def visit_ListComp(self, comp: ast.ListComp):
    #    """
    #    Convert a Python list comprehension expression AST node into a Qwerty
    #    ``RepeatTensor`` AST node. For example, this code:

    #        ['0' for i in range(5)]

    #    is converted to a RepeatTensor AST node whose child is a QubitLiteral.
    #    (This particular example is equivalent to '00000'.) Here, ``range`` is
    #    a keyword whose operand is a dimension variable expression.
    #    """
    #    return RepeatTensor(*self.list_comp_helper(comp))

    def visit_List(self, list_: ast.List):
        """
        Convert an empty Python list literal AST node into a Qwerty
        ``UnitLiteral`` AST node. That is, ``[]`` becomes an
        ``qpu::Expr::UnitLiteral``.
        """
        dbg = self.get_debug_loc(list_)
        if list_.elts:
            raise QwertySyntaxError('Python list literals are not supported '
                                    '(except for the Qwerty unit literal '
                                    '`[]`).', dbg)
        return QpuExpr.new_unit_literal(dbg)

    def visit_Call(self, call: ast.Call) -> QpuExpr:
        """
        As syntactic sugar, convert a Python call expression into a ``Pipe``
        Qwerty AST node. In general, the shorthand::

            f(arg1,arg2,...,argn)

        is equivalent to::

            (arg1,arg2,...,argn) | f

        There is also unrelated special handling for ``bit[4](0b1101)``.
        """
        if call.keywords:
            raise QwertySyntaxError('Keyword arguments not supported in '
                                    'call', self.get_debug_loc(call))

        dbg = self.get_debug_loc(call)

        # Handling for `bit[4](0b1101)`, bit literals
        if self.is_bit_literal(call):
            return self.extract_bit_literal(call)
        elif self.is_vector_atom_intrinsic(call):
            return self.extract_qubit_literal(call)
        elif isinstance(name_node := call.func, ast.Name) \
                and (intrinsic_name := name_node.id) in INTRINSICS:
            expected_num_args = INTRINSICS[intrinsic_name]
            args = call.args
            actual_num_args = len(args)
            if actual_num_args != expected_num_args:
                raise QwertySyntaxError('Wrong number of arguments to intrinsic '
                                        f'{intrinsic_name}(): expected '
                                        f'{expected_num_args} arguments, got '
                                        f'{actual_num_args}.', dbg)
            if intrinsic_name == '__MEASURE__':
                basis_node, = args
                basis = self.extract_basis(basis_node)
                return QpuExpr.new_measure(basis, dbg)
            elif intrinsic_name == '__DISCARD__':
                return QpuExpr.new_discard(dbg)
            elif intrinsic_name in EMBED_KINDS:
                func_arg, = args
                func = self.visit(func_arg)
                embed_kind = EMBED_KINDS[intrinsic_name]
                return QpuExpr.new_embed_classical(func, embed_kind, dbg)
            else:
                assert False, "compiler bug: intrinsic parsing misconfigured"
        else:
            rhs = self.visit(call.func)

            if not call.args:
                lhs = QpuExpr.new_unit_literal(dbg)
            elif len(call.args) == 1:
                lhs = self.visit(call.args[0])
            else:
                n_args = len(call.args)
                raise QwertySyntaxError('Either one or zero arguments can be '
                                        'passed to a function call, but got '
                                        f'{n_args} arguments.', dbg)

            return QpuExpr.new_pipe(lhs, rhs, dbg)

    #def visit_Tuple(self, tuple_: ast.Tuple):
    #    """
    #    Convert a Python tuple literal into a Qwerty tuple literal. Trust me,
    #    this one is thrilling.
    #    """
    #    dbg = self.get_debug_loc(tuple_)
    #    elts = self.visit(tuple_.elts)
    #    return TupleLiteral(dbg, elts)

    def visit_Attribute(self, attr: ast.Attribute):
        """
        Convert a Python attribute AST node into a metaQwerty AST
        ``BasisMacro`` or ``ExprMacro`` node.
        """
        dbg = self.get_debug_loc(attr)
        arg = attr.value
        name = attr.attr

        if isinstance(name_arg := arg, ast.Name) \
                and (arg_captured := self.capture_expr(
                        name_arg.id, self.get_debug_loc(name_arg))) \
                    is not None:
            return QpuExpr.new_expr_macro(name, arg_captured, dbg)
        else:
            try:
                arg_basis = self.extract_basis(arg)
            # TODO: do a more granular catch here
            except QwertySyntaxError:
                arg_expr = self.visit(arg)
                return QpuExpr.new_expr_macro(name, arg_expr, dbg)
            else:
                return QpuExpr.new_basis_macro(name, arg_basis, dbg)

    def visit_IfExp(self, if_expr: ast.IfExp):
        """
        Convert a Python conditional expression AST node into a Qwerty
        classical branching AST node. For example, ``x if y or z`` becomes a
        Qwerty ``Conditional`` AST node with three children.
        """
        dbg = self.get_debug_loc(if_expr)
        then_expr = self.visit(if_expr.body)
        else_expr = self.visit(if_expr.orelse)

        try:
            # Ban basis aliases because we do not want to interpret
            # ``flip if my_bit else id`` as a predication. However, still allow
            # singleton vectors as in ``flip if '1_' else id`` because that
            # wouldn't type check anyway (a qubit is not a bit!).
            pred_basis = self.extract_basis(if_expr.test,
                                            allow_singleton=True,
                                            allow_alias=False)
        # TODO: do a more granular catch here
        except QwertySyntaxError:
            cond_expr = self.visit(if_expr.test)
            return QpuExpr.new_conditional(then_expr, else_expr, cond_expr, dbg)
        else:
            return QpuExpr.new_predicated(then_expr, else_expr, pred_basis, dbg)

    def visit_Compare(self, compare: ast.Compare):
        """
        Convert a Python `x in y` AST node into a Qwerty predication
        `x if y else id**N`.
        """
        dbg = self.get_debug_loc(compare)

        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            raise QwertySyntaxError('Invalid comparison syntax', dbg)

        left_node = compare.left
        op, = compare.ops
        right_node, = compare.comparators

        if not isinstance(op, ast.In):
            cmp_name = type(op).__name__
            raise QwertySyntaxError(f'Unknown comparison {cmp_name}', dbg)

        if self._func_name is None:
            raise QwertySyntaxError('Cannot use `in` syntax outside of a '
                                    'function', dbg)
        else:
            internal_dim_var = self.allocate_internal_dim_var(dbg)

        # '?'**N
        pad_vec = Vector.new_vector_broadcast_tensor(
            Vector.new_pad_vector(dbg),
            internal_dim_var,
            dbg)
        pad_basis = Basis.new_basis_literal([pad_vec], dbg)
        # '?'**N >> '?'**N
        identity = QpuExpr.new_basis_translation(pad_basis, pad_basis, dbg)

        then_func = self.visit(left_node)
        else_func = identity
        pred_basis = self.extract_basis(right_node)
        return QpuExpr.new_predicated(then_func, else_func, pred_basis, dbg)

    #def visit_BoolOp(self, boolOp: ast.BoolOp):
    #    if isinstance(boolOp.op, ast.Or):
    #        return self.visit_BoolOp_Or(boolOp)
    #    else:
    #        op_name = type(boolOp.op).__name__
    #        raise QwertySyntaxError('Unknown boolean operation {}'
    #                                .format(op_name),
    #                                self.get_debug_loc(boolOp))

    #def visit_BoolOp_Or(self, boolOp: ast.BoolOp):
    #    """
    #    Convert a Python Boolean expression with an ``or`` into a Qwerty
    #    superposition AST node. For example, ``0.25*'0' or 0.75*'1'`` becomes a
    #    ``SuperpositionLiteral`` AST node with two children.
    #    """
    #    dbg = self.get_debug_loc(boolOp)
    #    operands = boolOp.values
    #    if len(operands) < 2:
    #        raise QwertySyntaxError('Superposition needs at least two operands', dbg)

    #    pairs = []
    #    had_prob = False

    #    for operand in operands:
    #        # Common case: 0.5 * '0'
    #        if isinstance(mult_binop := operand, ast.BinOp) \
    #                and isinstance(mult_binop.op, ast.Mult):
    #            prob_node = mult_binop.left
    #            vec_node = mult_binop.right
    #        # Deal with some operator precedence aggravation: for 0.5 * '0'@45
    #        # the root node is actually @. Rearrange this for programmer
    #        # convenience
    #        elif isinstance(matmult_binop := operand, ast.BinOp) \
    #                 and isinstance(matmult_binop.op, ast.MatMult) \
    #                 and isinstance(mult_binop := matmult_binop.left, ast.BinOp) \
    #                 and isinstance(mult_binop.op, ast.Mult):
    #            prob_node = mult_binop.left
    #            vec_node = ast.BinOp(mult_binop.right, ast.MatMult(),
    #                                 matmult_binop.right, lineno=matmult_binop.lineno,
    #                                 col_offset=matmult_binop.col_offset)
    #        else:
    #            prob_node = None
    #            vec_node = operand

    #        has_prob = prob_node is not None

    #        if pairs and has_prob ^ had_prob:
    #            operand_dbg = self.get_debug_loc(operand)
    #            raise QwertySyntaxError(
    #                'Either all operands of a superposition operator should '
    #                'have explicit probabilities, or none should have '
    #                'explicit probabilities', operand_dbg)

    #        had_prob = has_prob

    #        if has_prob:
    #            if not isinstance(prob_node, ast.Constant):
    #                prob_dbg = self.get_debug_loc(prob_node)
    #                raise QwertySyntaxError(
    #                    'Currently, probabilities in a superposition literal '
    #                    'must be integer constants', prob_dbg)
    #            prob_const_node = prob_node
    #            prob_val = prob_const_node.value

    #            if not isinstance(prob_val, float) \
    #                    and not isinstance(prob_val, int):
    #                prob_dbg = self.get_debug_loc(prob_node)
    #                raise QwertySyntaxError(
    #                    'Probabilities in a superposition literal must be '
    #                    'floats, not {}'.format(str(type(prob_val))), prob_dbg)
    #        else:
    #            prob_val = 1.0 / len(operands)

    #        pair = (prob_val, self.visit(vec_node))
    #        pairs.append(pair)

    #    return SuperposLiteral(dbg, pairs)

    def visit(self, node: ast.AST):
        if isinstance(node, ast.Constant):
            return self.visit_Constant(node)
        elif isinstance(node, ast.Subscript):
            return self.visit_Subscript(node)
        #elif isinstance(node, ast.BinOp):
        elif isinstance(node, ast.BinOp):
            return self.visit_BinOp(node)
        #elif isinstance(node, ast.UnaryOp):
        #    return self.visit_UnaryOp(node)
        elif isinstance(node, ast.Set):
            return self.visit_Set(node)
        elif isinstance(node, ast.GeneratorExp):
            return self.visit_GeneratorExp(node)
        #elif isinstance(node, ast.ListComp):
        #    return self.visit_ListComp(node)
        elif isinstance(node, ast.List):
            return self.visit_List(node)
        elif isinstance(node, ast.Call):
            return self.visit_Call(node)
        #elif isinstance(node, ast.Tuple):
        #    return self.visit_Tuple(node)
        elif isinstance(node, ast.Attribute):
            return self.visit_Attribute(node)
        elif isinstance(node, ast.IfExp):
            return self.visit_IfExp(node)
        elif isinstance(node, ast.Compare):
            return self.visit_Compare(node)
        #elif isinstance(node, ast.BoolOp):
        #    return self.visit_BoolOp(node)
        else:
            return self.base_visit(node)

class QpuPreludeVisitor(QpuVisitor):
    """
    For parsing ``@qpu_prelude``s.
    """

    def __init__(self, filename: str = '', line_offset: int = 0,
                 col_offset: int = 0):
        super().__init__(trivial_name_generator(), TrivialCapturer(), filename,
                         line_offset, col_offset)

    def visit_FunctionDef(self, func_def: ast.FunctionDef) -> QpuPrelude:
        """
        Overload ``QpuVisitor.visit_FunctionDef()`` to parse the simpler
        prelude function AST structure.
        """
        dbg = self.get_debug_loc(func_def)

        # Sanity check for unsupported features
        for arg_prop in ('args', 'posonlyargs', 'vararg', 'kwonlyargs',
                         'kw_defaults', 'kwarg', 'defaults'):
            if getattr(func_def.args, arg_prop):
                raise QwertySyntaxError('Preludes cannot have arguments', dbg)

        if (n_decorators := len(func_def.decorator_list)) != 1:
            raise QwertySyntaxError('Wrong number of decorators ({} > 2) '
                                    'for prelude'.format(n_decorators), dbg)
        else:
            decorator, = func_def.decorator_list
            if not isinstance(decorator, ast.Name) \
                    or decorator.id != 'qpu_prelude':
                raise QwertySyntaxError('Unknown decorator for prelude', dbg)

        if func_def.returns:
            raise QwertySyntaxError('Preludes may not have a return type.',
                                    dbg)

        body = self.visit(func_def.body)
        return QpuPrelude(body, dbg)

def convert_qpu_func_ast(module: ast.Module,
                         name_generator: Callable[[str], str],
                         capturer: Capturer, filename: str = '',
                         line_offset: int = 0, col_offset: int = 0) \
                        -> QpuFunctionDef:
    """
    Run the ``QpuVisitor`` on the provided Python AST to convert to a Qwerty
    ``@qpu`` AST and return the result. The return value is the same as
    ``convert_func_ast()`` above.
    """
    if not isinstance(module, ast.Module):
        raise QwertySyntaxError('Expected top-level Module node in Python AST',
                                None) # This should not happen

    visitor = QpuVisitor(name_generator, capturer, filename, line_offset,
                         col_offset)
    return visitor.visit_Module(module)

def convert_qpu_prelude_ast(module: ast.Module, filename: str = '',
                            line_offset: int = 0, col_offset: int = 0) \
                           -> QpuPrelude:
    """
    Run the ``QpuPreludeVisitor`` on the provided Python AST to convert to a
    Qwerty ``@qpu`` prelude.
    """
    if not isinstance(module, ast.Module):
        raise QwertySyntaxError('Expected top-level Module node in Python AST',
                                None) # This should not happen

    visitor = QpuPreludeVisitor(filename, line_offset, col_offset)
    return visitor.visit_Module(module)

def convert_qpu_repl_input(root: ast.Interactive, *, capturer=None) -> QpuStmt:
    """
    Convert a line from the Qwerty REPL into a Qwerty AST. Always returns a
    statement.
    """

    if not isinstance(root, ast.Interactive):
        raise QwertySyntaxError('Expected top-level Interactive node in '
                                'Python AST')
    if len(root.body) != 1:
        raise QwertySyntaxError('Expected one statement as input, not '
                                f'{len(root.body)} statements')

    stmt, = root.body
    capturer = capturer or TrivialCapturer()
    visitor = QpuVisitor(name_generator=trivial_name_generator(),
                         capturer=capturer,
                         filename='<input>',
                         line_offset=0,
                         col_offset=0)
    return visitor.visit(stmt)

#################### @CLASSICAL DSL ####################

class ClassicalVisitor(BaseVisitor):
    """
    Python AST visitor for syntax specific to ``@classical`` kernels.
    """

    def __init__(self, name_generator: Callable[[str], str],
                 capturer: Capturer, filename: str = '', line_offset: int = 0,
                 col_offset: int = 0):
        super().__init__(ClassicalExpr, ClassicalStmt, ClassicalFunctionDef,
                         name_generator, capturer, filename, line_offset,
                         col_offset)

    def visit_FunctionDef(self, func_def: ast.FunctionDef) -> ClassicalFunctionDef:
        """
        Convert a ``@classical`` kernel into a Qwerty AST node.
        """
        return super().base_visit_FunctionDef(func_def, 'classical')

#    def visit_Name(self, name: ast.Name):
#        """
#        Convert a Python AST identitifer node into a Qwerty variable name AST
#        node. (The ``@classical`` DSL does not have reserved keywords.)
#        """
#        var_name = name.id
#        dbg = self.get_debug_loc(name)
#        return Variable(dbg, var_name)
#
    def visit_UnaryOp(self, unaryOp: ast.UnaryOp) -> ClassicalExpr:
        if isinstance(unaryOp.op, ast.Invert):
            return self.visit_UnaryOp_Invert(unaryOp)
        else:
            op_name = type(unaryOp.op).__name__
            raise QwertySyntaxError('Unknown unary operation {}'
                                    .format(op_name),
                                    self.get_debug_loc(unaryOp))

    def visit_UnaryOp_Invert(self, unaryOp: ast.UnaryOp) -> ClassicalExpr:
        """
        Convert a Python bitwise complement AST node into the same thing in the
        Qwerty AST. For example, ``~x`` becomes a ``BitUnaryOp`` Qwerty AST
        node.
        """
        operand = self.visit(unaryOp.operand)
        dbg = self.get_debug_loc(unaryOp)
        return ClassicalExpr.new_unary_op(UnaryOpKind.Not, operand, dbg)

    def visit_BinOp(self, binOp: ast.BinOp):
        if isinstance(binOp.op, ast.BitAnd):
            return self.visit_BinOp_BitAnd(binOp)
        elif isinstance(binOp.op, ast.BitXor):
            return self.visit_BinOp_BitXor(binOp)
        elif isinstance(binOp.op, ast.BitOr):
            return self.visit_BinOp_BitOr(binOp)
        elif isinstance(binOp.op, ast.Mod):
            return self.visit_BinOp_Mod(binOp)
        else:
            op_name = type(binOp.op).__name__
            raise QwertySyntaxError('Unknown binary operation {}'
                                    .format(op_name),
                                    self.get_debug_loc(binOp))

    def visit_BinOp_BitAnd(self, binOp: ast.BinOp):
        """
        Convert a Python binary bitwise AND expression into the same thing in
        the Qwerty AST. For example, ``x & y`` becomes a ``BitBinaryOp`` Qwerty
        AST node.
        """
        left = self.visit(binOp.left)
        right = self.visit(binOp.right)
        dbg = self.get_debug_loc(binOp)
        return ClassicalExpr.new_binary_op(BinaryOpKind.And, left, right, dbg)

    def visit_BinOp_BitXor(self, binOp: ast.BinOp):
        """
        Convert a Python binary bitwise XOR expression into the same thing in
        the Qwerty AST. For example, ``x ^ y`` becomes a ``BitBinaryOp`` Qwerty
        AST node.
        """
        left = self.visit(binOp.left)
        right = self.visit(binOp.right)
        dbg = self.get_debug_loc(binOp)
        return ClassicalExpr.new_binary_op(BinaryOpKind.Xor, left, right, dbg)

    def visit_BinOp_BitOr(self, binOp: ast.BinOp):
        """
        Convert a Python binary bitwise OR expression into the same thing in
        the Qwerty AST. For example, ``x | y`` becomes a ``BitBinaryOp`` Qwerty
        AST node.
        """
        left = self.visit(binOp.left)
        right = self.visit(binOp.right)
        dbg = self.get_debug_loc(binOp)
        return ClassicalExpr.new_binary_op(BinaryOpKind.Or, left, right, dbg)

    def visit_BinOp_Mod(self, binOp: ast.BinOp):
        """
        Convert a Python expression ``x % N`` to a Qwerty ``Mod`` AST node.
        This is useful for making a simple period finding oracle.
        This can also convert a Python expression ``X**2**J*y % N`` to a Qwerty
        ``ModMulOp`` AST node. This is useful for the order finding oracle in
        the order finding subroutine of Shor's algorithm.
        """
        dbg = self.get_debug_loc(binOp)

        # Modular multiplication: X**2**J*y % N
        if isinstance((left_mul := binOp.left), ast.BinOp) \
                and isinstance(left_mul.op, ast.Mult) \
                and isinstance((pow_ := left_mul.left), ast.BinOp) \
                and isinstance(pow_.op, ast.Pow) \
                and isinstance((inner_pow := pow_.right), ast.BinOp) \
                and isinstance(inner_pow.op, ast.Pow):
            x = self.extract_dimvar_expr(pow_.left)
            exp_base_node = inner_pow.left
            j = self.extract_dimvar_expr(inner_pow.right)
            y = self.visit(left_mul.right)
            modN = self.extract_dimvar_expr(binOp.right)

            if not isinstance(exp_base_node, ast.Constant):
                raise QwertySyntaxError('Dimvars not allowed in base of exponent in '
                                        'modular multiplication', dbg)
            exp_base = exp_base_node.value
            if exp_base != 2:
                raise QwertySyntaxError('Currently, only 2 is supported as the '
                                        'base of the exponent in modular '
                                        'multiplication', dbg)

            return ClassicalExpr.new_mod_mul(x, j, y, modN, dbg)
        # Ordinary modulus x % 4
        else:
            dividend = self.visit(binOp.left)
            divisor = self.extract_dimvar_expr(binOp.right)

            return ClassicalExpr.new_mod(dividend, divisor, dbg)
#
    def visit_Call(self, call: ast.Call):
        """
        Convert a Python call expression into either a ``BitLiteral`` Qwerty
        AST node (for e.g. ``bit[4](0b1101)``) or other bitwise operations
        expressed as (pseudo)functions in Python syntax.

        For example, ``x.repeat(N)`` is converted to a ``BitRepeat`` Qwerty AST
        node with one child ``x``; ``x.rotl(y)`` is converted to an appropriate
        ``BitBinaryOp`` node with two children; and ``x.xor_reduce()`` is
        converted to a ``BitReduceOp`` node with one child.
        """
        if self.is_bit_literal(call):
            return self.extract_bit_literal(call)
        else: # xor_reduce(), and_reduce(), etc
            func = call.func
            if not isinstance(func, ast.Attribute):
                raise QwertySyntaxError('I expect function calls to be of the '
                                        'form expression.FUNC(), but this call '
                                        'is not',
                                        self.get_debug_loc(call))
            attr = func
            operand = attr.value
            func_name = attr.attr

            reduce_pseudo_funcs = {'xor_reduce': BinaryOpKind.Xor,
                                   'and_reduce': BinaryOpKind.And,
                                   'or_reduce': BinaryOpKind.Or}
            #binary_pseudo_funcs = {'rotr': BIT_ROTR,
            #                       'rotl': BIT_ROTL}
            if func_name in reduce_pseudo_funcs:
                if call.args or call.keywords:
                    raise QwertySyntaxError('Arguments cannot be passed to a '
                                            'reduction pseudo-function',
                                            self.get_debug_loc(call))
                dbg = self.get_debug_loc(call)
                kind = reduce_pseudo_funcs[func_name]
                val = self.visit(operand)
                return ClassicalExpr.new_reduce_op(kind, val, dbg)
            #elif func_name in binary_pseudo_funcs:
            #    if call.keywords:
            #        raise QwertySyntaxError('Keywords arguments not '
            #                                'supported to {}()'
            #                                .format(func_name),
            #                                self.get_debug_loc(call))
            #    if len(call.args) != 1:
            #        raise QwertySyntaxError('{}() expects one positional '
            #                                'argument: the rotation amount'
            #                                .format(func_name),
            #                                self.get_debug_loc(call))
            #    val = self.visit(operand)
            #    amount_node = call.args[0]
            #    amount = self.visit(amount_node)
            #    dbg = self.get_debug_loc(call)
            #    return BitBinaryOp(dbg, binary_pseudo_funcs[func_name], val,
            #                       amount)
            #elif func_name == 'repeat':
            #    if call.keywords:
            #        raise QwertySyntaxError('Keywords arguments not '
            #                                'supported to {}()'
            #                                .format(func_name),
            #                                self.get_debug_loc(call))
            #    if len(call.args) != 1:
            #        raise QwertySyntaxError('{}() expects one positional '
            #                                'argument: the amount of times to '
            #                                'repeat'
            #                                .format(func_name),
            #                                self.get_debug_loc(call))
            #    bits = self.visit(operand)
            #    amount_node = call.args[0]
            #    amount = self.extract_dimvar_expr(amount_node)
            #    dbg = self.get_debug_loc(call)
            #    return BitRepeat(dbg, bits, amount)
            else:
                raise QwertySyntaxError('Unknown pseudo-function {}'
                                        .format(func_name),
                                        self.get_debug_loc(call))
#
#    def visit_Tuple(self, tup: ast.Tuple):
#        """
#        Convert a Python tuple literal to a nest of Qwerty ``BitConcat`` AST
#        nodes.
#        """
#        if not tup.elts:
#            raise QwertySyntaxError('Empty tuple not supported',
#                                    self.get_debug_loc(tup))
#        cur = self.visit(tup.elts[0])
#        for elt in tup.elts[1:]:
#            dbg = self.get_debug_loc(tup)
#            cur = BitConcat(dbg, cur, self.visit(elt))
#        return cur

    def visit_Subscript(self, sub: ast.Subscript):
        """
        Convert a Python getitem expression to Qwerty ``Slice`` AST node.
        """
        dbg = self.get_debug_loc(sub)
        val = self.visit(sub.value)
        if isinstance(sub.slice, ast.Slice):
            if sub.slice.step is not None:
                raise QwertySyntaxError('[:::] syntax not supported', dbg)

            if sub.slice.lower is None:
                raise QwertySyntaxError('Start index of slice required', dbg)
            else:
                lower = self.extract_dimvar_expr(sub.slice.lower)

            if sub.slice.upper is None:
                raise QwertySyntaxError('End index of slice required', dbg)
            else:
                upper = self.extract_dimvar_expr(sub.slice.upper)
        else:
            # [lower:lower+1]
            lower = self.extract_dimvar_expr(sub.slice)
            upper = DimExpr.new_sum(lower, DimExpr.new_const(1, dbg), dbg)

        return ClassicalExpr.new_slice(val, lower, upper, dbg)

    def visit(self, node: ast.AST):
        if isinstance(node, ast.UnaryOp):
            return self.visit_UnaryOp(node)
        elif isinstance(node, ast.BinOp):
            return self.visit_BinOp(node)
        elif isinstance(node, ast.Call):
            return self.visit_Call(node)
        #elif isinstance(node, ast.Tuple):
        #    return self.visit_Tuple(node)
        elif isinstance(node, ast.Subscript):
            return self.visit_Subscript(node)
        else:
            return self.base_visit(node)

def convert_classical_func_ast(module: ast.Module,
                               name_generator: Callable[[str], str],
                               capturer: Capturer, filename: str = '',
                               line_offset: int = 0, col_offset: int = 0) \
                              -> ClassicalFunctionDef:
    """
    Run the ``ClassicalVisitor`` on the provided Python AST to convert to a
    Qwerty ``@classical`` AST and return the result. The return value is the
    same as ``convert_func_ast()`` above.
    """
    if not isinstance(module, ast.Module):
        raise QwertySyntaxError('Expected top-level Module node in Python AST',
                                None) # This should not happen

    visitor = ClassicalVisitor(name_generator, capturer, filename, line_offset,
                               col_offset)
    return visitor.visit_Module(module)

def convert_classical_repl_input(root: ast.Interactive, *, capturer=None) -> ClassicalStmt:
    """
    Convert a line from a hypothetical Qwerty ``@classical`` REPL into a Qwerty
    classical AST. Always returns a statement. Right now, this is only used in
    unit tests.
    """

    if not isinstance(root, ast.Interactive):
        raise QwertySyntaxError('Expected top-level Interactive node in '
                                'Python AST')
    if len(root.body) != 1:
        raise QwertySyntaxError('Expected one statement as input, not '
                                f'{len(root.body)} statements')

    stmt, = root.body
    capturer = capturer or TrivialCapturer()
    visitor = ClassicalVisitor(name_generator=lambda name: name,
                               capturer=capturer,
                               filename='<input>',
                               line_offset=0,
                               col_offset=0)
    return visitor.visit(stmt)
