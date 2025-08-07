"""
Convert a Python AST to a Qwerty AST by recognizing patterns in the Python AST
formed by Qwerty syntax.
"""

import ast
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import List, Tuple, Union, Optional
from .runtime import bit
from .err import EXCLUDE_ME_FROM_STACK_TRACE_PLEASE, QwertySyntaxError, \
                 get_frame, set_dbg_frame
from ._qwerty_pyrt import DebugLoc, RegKind, Type, QpuFunctionDef, \
                          ClassicalFunctionDef, QLit, Vector, Basis, \
                          QpuStmt, ClassicalStmt, QpuExpr, ClassicalExpr, \
                          BasisGenerator, UnaryOpKind, BinaryOpKind, EmbedKind

#################### COMMON CODE FOR BOTH @QPU AND @CLASSICAL DSLs ####################

class AstKind(Enum):
    QPU = 1
    CLASSICAL = 2

class CapturedValue(ABC):
    """A captured value."""
    @abstractmethod
    def to_expr(self, expr_class: type[QpuExpr | ClassicalExpr],
                dbg: Optional[DebugLoc]) -> QpuExpr | ClassicalExpr:
        ...

class CapturedSymbol(CapturedValue):
    """A captured (and mangled) function name."""

    def __init__(self, mangled_name: str):
        self.mangled_name = mangled_name

    def to_expr(self, expr_class: type[QpuExpr | ClassicalExpr],
                dbg: Optional[DebugLoc]) -> QpuExpr | ClassicalExpr:
        return expr_class.new_variable(self.mangled_name, dbg)

class CapturedBitReg(CapturedValue):
    """A captured bit[N]."""

    def __init__(self, bit_reg: bit):
        self.bit_reg = bit_reg

    def to_expr(self, expr_class: type[QpuExpr | ClassicalExpr],
                dbg: Optional[DebugLoc]) -> QpuExpr | ClassicalExpr:
        return expr_class.new_bit_literal(int(self.bit_reg),
                                          self.bit_reg.n_bits,
                                          dbg)

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

def convert_ast(ast_kind: AstKind, module: ast.Module,
                name_generator: Callable[[str], str], capturer: Capturer,
                filename: str = '', line_offset: int = 0,
                col_offset: int = 0) -> QpuFunctionDef | ClassicalFunctionDef:
    """
    Take in a Python AST for a function parsed with ``ast.parse(mode='exec')``
    and return a ``Kernel`` Qwerty AST node.

    The ``line_offset`` and `col_offset`` are useful (respectively) because a
    ``@qpu``/``@classical`` kernel may begin after the first line of the file,
    and the caller may de-indent source code to avoid angering ``ast.parse()``.
    """
    if ast_kind == AstKind.QPU:
        return convert_qpu_ast(module, name_generator, capturer, filename,
                               line_offset, col_offset)
    elif ast_kind == AstKind.CLASSICAL:
        return convert_classical_ast(module, name_generator, capturer,
                                     filename, line_offset, col_offset)
    else:
        raise ValueError('unknown AST type {}'.format(ast_kind))

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

    #def extract_dimvar_expr(self, node: ast.AST) -> DimVarExpr:
    #    """
    #    Return a dimension variable expression given a subtree of the AST, for
    #    example::

    #        std[2*N + 1]
    #            ^^^^^^^
    #    """

    #    # TODO: support more advanced constant expressions
    #    if isinstance(node, ast.Name):
    #        dimvar_name = node.id
    #        if dimvar_name in self.dim_vars:
    #            return DimVarExpr(dimvar_name, 0)
    #        else:
    #            raise QwertySyntaxError('Unknown type variable {} found in '
    #                                    'constant type expression'
    #                                    .format(dimvar_name),
    #                                    self.get_debug_loc(node))
    #    elif isinstance(node, ast.Constant) \
    #            and isinstance(node.value, int):
    #        return DimVarExpr('', node.value)
    #    elif isinstance(node, ast.BinOp) \
    #            and isinstance(node.op, ast.Add):
    #        left = self.extract_dimvar_expr(node.left)
    #        left += self.extract_dimvar_expr(node.right)
    #        return left
    #    elif isinstance(node, ast.BinOp) \
    #            and isinstance(node.op, ast.Sub):
    #        left = self.extract_dimvar_expr(node.left)
    #        left -= self.extract_dimvar_expr(node.right)
    #        return left
    #    elif isinstance(node, ast.BinOp) \
    #            and isinstance(node.op, ast.Mult):
    #        left = self.extract_dimvar_expr(node.left)
    #        right = self.extract_dimvar_expr(node.right)

    #        if not left.is_constant() and not right.is_constant():
    #            raise QwertySyntaxError('Dimvars, or constant type '
    #                                    'expressions containing dimvars, '
    #                                    'cannot be multiplied together',
    #                                    self.get_debug_loc(node))
    #        left *= right
    #        return left
    #    else:
    #        raise QwertySyntaxError('Unsupported constant type expression',
    #                                self.get_debug_loc(node))

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
            if type_name == 'qubit':
                return Type.new_reg(RegKind.Qubit, 1)
            elif type_name == 'bit':
                return Type.new_reg(RegKind.Bit, 1)
            elif type_name == 'qfunc':
                return Type.new_func(Type.new_reg(RegKind.Qubit, 1),
                                     Type.new_reg(RegKind.Qubit, 1))
            elif type_name == 'rev_qfunc':
                return Type.new_rev_func(Type.new_reg(RegKind.Qubit, 1))
            else:
                raise QwertySyntaxError('Unknown type name {} found'
                                        .format(type_name),
                                        self.get_debug_loc(node))
        elif isinstance(node, ast.Subscript) \
                and isinstance(node.value, ast.Name) \
                and (type_name := node.value.id) in ('bit', 'qubit', 'qfunc', 'rev_qfunc') \
                and isinstance((dim_node := node.slice), ast.Constant) \
                and isinstance((dim := dim_node.value), int):
            if type_name == 'qubit':
                return Type.new_reg(RegKind.Qubit, dim)
            elif type_name == 'bit':
                return Type.new_reg(RegKind.Bit, dim)
            elif type_name == 'qfunc':
                return Type.new_func(Type.new_reg(RegKind.Qubit, dim),
                                     Type.new_reg(RegKind.Qubit, dim))
            elif type_name == 'rev_qfunc':
                return Type.new_rev_func(Type.new_reg(RegKind.Qubit, dim),
                                         Type.new_reg(RegKind.Qubit, dim))
            else:
                raise QwertySyntaxError('Unknown type name {}[N] found'
                                        .format(type_name),
                                        self.get_debug_loc(node))
        elif isinstance(node, ast.Subscript) \
                and isinstance(node.value, ast.Name) \
                and (type_name := node.value.id) == 'qfunc' \
                and isinstance((dims_node := node.slice), ast.Tuple) \
                and len(dims_node_elts := dims_node.elts) != 2 \
                and isinstance((in_dim_node := dims_node_elts[0]), ast.Constant) \
                and isinstance((out_dim_node := dims_node_elts[1]), ast.Constant) \
                and isinstance((in_dim := in_dim_node.value), int) \
                and isinstance((out_dim := out_dim_node.value), int):
            return Type.new_func(Type.new_reg(RegKind.Qubit, in_dim),
                                 Type.new_reg(RegKind.Qubit, out_dim))
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
        dbg = self.get_debug_loc(expr_stmt)
        expr = self.visit(expr_stmt.value)
        return self._stmt_class.new_expr(expr, dbg)

    def base_visit_FunctionDef(self, func_def: ast.FunctionDef,
                               decorator_name: str) \
                              -> QpuFunctionDef | ClassicalFunctionDef:
        """
        Common code for processing the function Python AST node for both
        ``@classical`` and ``@qpu`` kernels.
        """
        func_name = func_def.name

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
            raise QwertySyntaxError('Mysterious non-@{}[[N,M]](capture1, '
                                    'capture2) decorator for {}()'
                                    .format(decorator_name,
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
            if not arg_type:
                raise QwertySyntaxError('Currently, type annotations are '
                                        'required for arguments such as {} '
                                        'argument of {}()'
                                        .format(arg_name, func_name),
                                        self.get_debug_loc(arg))

            actual_arg_type = self.extract_type_literal(arg_type)
            args.append((actual_arg_type, arg_name))

        # Now, figure out return type
        if not func_def.returns:
            raise QwertySyntaxError('Currently, type annotations are '
                                    'required for functions such as {}() '
                                    .format(func_name),
                                    self.get_debug_loc(func_def))
        ret_type = self.extract_type_literal(func_def.returns)

        # Great, now we have everything we need to build the AST node...
        dbg = self.get_debug_loc(func_def)

        # ...except traversing the function body
        body = self.visit(func_def.body)

        generated_func_name = self.name_generator(func_name)
        return self._func_class(generated_func_name, args, ret_type, body, is_rev, dbg)

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

    def visit_Assign(self, assign: ast.Assign) -> QpuStmt:
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
            expr = self.visit(assign.value)
            var_name = tgt.id

            if self.capturer.shadows_python_variable(var_name):
                raise QwertySyntaxError('Cannot define a variable '
                                        f'({var_name}) that shadows a Python '
                                        'variable.', dbg)

            return self._stmt_class.new_assign(var_name, expr, dbg)
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

    def visit_Name(self, name: ast.Name) -> QpuExpr | ClassicalExpr:
        """
        Convert a Python AST identitifer node into a Qwerty variable name AST
        node. For example, ``foobar`` becomes a Qwerty ``Variable`` AST node.
        """
        var_name = name.id
        dbg = self.get_debug_loc(name)
        try:
            captured = self.capturer.capture(var_name)
        except CaptureError as err:
            var_type_name = err.type_name
            raise QwertySyntaxError(f'The Python object named {var_name} is '
                                    'referenced, but it has Python type '
                                    f'{var_type_name}, which cannot be used '
                                    'in Qwerty kernels.', dbg)

        if captured is None:
            return self._expr_class.new_variable(var_name, dbg)
        else:
            return captured.to_expr(self._expr_class, dbg)

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

class QpuVisitor(BaseVisitor):
    """
    Python AST visitor for syntax specific to ``@qpu`` kernels.
    """

    def __init__(self, name_generator: Callable[[str], str],
                 capturer: Capturer, filename: str = '', line_offset: int = 0,
                 col_offset: int = 0):
        super().__init__(QpuExpr, QpuStmt, QpuFunctionDef, name_generator,
                         capturer, filename, line_offset, col_offset)

    def extract_qubit_literal(self, node: ast.AST) -> QLit:
        bv = self.extract_basis_vector(node)
        qlit = bv.convert_to_qubit_literal()
        if qlit is None:
            dbg = self.get_debug_loc(node)
            raise QwertySyntaxError(f"The symbols '?' and '_' are not allowed "
                                    "in qubit literals.", dbg)
        else:
            return qlit

    def convert_vector_atom(self, sym: str, dbg: DebugLoc) -> QLit:
        if sym == '0':
            return Vector.new_zero_vector(dbg)
        elif sym == '1':
            return Vector.new_one_vector(dbg)
        elif sym == '?':
            return Vector.new_pad_vector(dbg)
        elif sym == '_':
            return Vector.new_target_vector(dbg)
        else:
            raise QwertySyntaxError(f'Unknown qubit symbol {sym}', dbg)

    def extract_basis_vector(self, node: ast.AST) -> Vector:
        dbg = self.get_debug_loc(node)

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.extract_basis_vector(node.left)
            right = self.extract_basis_vector(node.right)
            return Vector.new_uniform_vector_superpos(left, right, dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            left = self.extract_basis_vector(node.left)
            right = self.extract_basis_vector(node.right)
            right_neg = Vector.new_vector_tilt(right, 180.0, dbg)
            return Vector.new_uniform_vector_superpos(left, right_neg, dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left = self.extract_basis_vector(node.left)
            right = self.extract_basis_vector(node.right)
            return Vector.new_vector_tensor([left, right], dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            q = self.extract_basis_vector(node.left)
            angle_deg = self.extract_float_const(node.right)
            return Vector.new_vector_tilt(q, angle_deg, dbg)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            q = self.extract_basis_vector(node.operand)
            return Vector.new_vector_tilt(q, 180.0, dbg)
        elif isinstance(node, ast.Constant) and node.value == '':
            return Vector.new_vector_unit(dbg)
        elif isinstance(node, ast.Constant) and len(node.value) == 1:
            return self.convert_vector_atom(node.value, dbg)
        elif isinstance(node, ast.Constant): # len(node.value) > 1
            return Vector.new_vector_tensor([self.convert_vector_atom(sym, dbg)
                                             for sym in node.value], dbg)
        else:
            node_name = type(node).__name__
            raise QwertySyntaxError('Unknown basis vector or qubit literal syntax {}'
                                    .format(node_name), dbg)

    def extract_basis_generator(self, node: ast.AST) -> BasisGenerator:
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
        else:
            node_name = type(node).__name__
            raise QwertySyntaxError('Unknown basis generator syntax {}'
                                    .format(node_name), dbg)

    def extract_basis(self, node: ast.AST) -> Basis:
        dbg = self.get_debug_loc(node)

        if isinstance(node, ast.Set) and not node.elts:
            return Basis.new_empty_basis_literal(dbg)
        elif isinstance(node, ast.Set) and node.elts:
            vecs = [self.extract_basis_vector(elt) for elt in node.elts]
            return Basis.new_basis_literal(vecs, dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left = self.extract_basis(node.left)
            right = self.extract_basis(node.right)
            return Basis.new_basis_tensor([left, right], dbg)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
            basis = self.extract_basis(node.left)
            gen = self.extract_basis_generator(node.right)
            return Basis.new_apply_basis_generator(basis, gen, dbg)
        elif (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)) \
                or (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            return Basis.new_basis_literal([self.extract_basis_vector(node)], dbg)
        else:
            node_name = type(node).__name__
            raise QwertySyntaxError('Unknown basis syntax {}'
                                    .format(node_name), dbg)

    # TODO: expand this into extract_float_expr()
    def extract_float_const(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) \
                and isinstance(node.value, (int, float)):
            return float(node.value)
        else:
            node_name = type(node).__name__
            raise QwertySyntaxError('Unsupported float constant syntax '
                                    + node_name, self.get_debug_loc(node))

    #def extract_float_expr(self, node: ast.AST):
    #    """
    #    Extract a float expression, like a tilt, for example::

    #        '1' @ (3*pi/2)
    #               ^^^^^^
    #    """
    #    dbg = self.get_debug_loc(node)

    #    if isinstance(node, ast.Name) and node.id == 'pi':
    #        return FloatLiteral(dbg, math.pi)
    #    if isinstance(node, ast.Name) and node.id in self.dim_vars:
    #        return FloatDimVarExpr(dbg, self.extract_dimvar_expr(node))
    #    elif isinstance(node, ast.Constant) \
    #            and type(node.value) in (int, float):
    #        return FloatLiteral(dbg, float(node.value))
    #    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
    #        return FloatNeg(dbg, self.extract_float_expr(node.operand))
    #    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
    #        return FloatBinaryOp(dbg, FLOAT_DIV,
    #                             self.extract_float_expr(node.left),
    #                             self.extract_float_expr(node.right))
    #    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
    #        return FloatBinaryOp(dbg, FLOAT_MUL,
    #                             self.extract_float_expr(node.left),
    #                             self.extract_float_expr(node.right))
    #    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
    #        return FloatBinaryOp(dbg, FLOAT_POW,
    #                             self.extract_float_expr(node.left),
    #                             self.extract_float_expr(node.right))
    #    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
    #        return FloatBinaryOp(dbg, FLOAT_MOD,
    #                             self.extract_float_expr(node.left),
    #                             self.extract_float_expr(node.right))
    #    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
    #        return FloatBinaryOp(dbg, FLOAT_ADD,
    #                             self.extract_float_expr(node.left),
    #                             self.extract_float_expr(node.right))
    #    elif isinstance(node, ast.Name):
    #        return self.visit(node)
    #    elif isinstance(node, ast.Subscript) \
    #            and isinstance(node.value, ast.Name) \
    #            and isinstance(node.slice, ast.List) \
    #            and len(node.slice.elts) == 1:
    #        val = self.visit(node.value)
    #        idx = node.slice.elts[0]
    #        lower = self.extract_dimvar_expr(idx)
    #        upper = lower.copy()
    #        upper += DimVarExpr('', 1)
    #        return Slice(dbg, val, lower, upper)
    #    else:
    #        node_name = type(node).__name__
    #        raise QwertySyntaxError('Unsupported float expression {}'
    #                                .format(node_name),
    #                                self.get_debug_loc(node))

    def visit_FunctionDef(self, func_def: ast.FunctionDef) -> QpuFunctionDef:
        """
        Convert a ``@qpu`` kernel into a ``QpuKernel`` Qwerty AST node.
        """
        return super().base_visit_FunctionDef(func_def, 'qpu')

    def visit_Constant(self, const: ast.Constant):
        value = const.value
        if isinstance(value, str):
            # A top-level string must be a qubit literal
            qlit = self.extract_qubit_literal(const)
            return QpuExpr.new_qlit(qlit)
        else:
            raise QwertySyntaxError('Unknown constant syntax',
                                    self.get_debug_loc(const))

    ## Broadcast tensor, i.e., tensoring something with itself repeatedly
    #def visit_Subscript(self, subscript: ast.Subscript):
    #    """
    #    Convert a Python getitem expression into a Qwerty ``BroadcastTensor``
    #    AST node. For example, the Python syntax ``t[N]`` becomes a
    #    ``BroadcastTensor`` with a child that is the conversion of ``t``.
    #    There is special-case handling here for ``fourier[N]`` and any other
    #    inseparable bases added in the future.
    #    """
    #    value, slice_ = subscript.value, subscript.slice
    #    dbg = self.get_debug_loc(subscript)

    #    if isinstance(list_ := slice_, ast.List) and list_.elts:
    #        value_node = self.visit(value)
    #        instance_vals = [self.extract_dimvar_expr(elt) for elt in list_.elts]
    #        return Instantiate(dbg, value_node, instance_vals)

    #    factor = self.extract_dimvar_expr(slice_)

    #    # Special case: Fourier basis
    #    if isinstance(value, ast.Name) \
    #            and value.id == 'fourier':
    #        return BuiltinBasis(dbg, FOURIER, factor)

    #    value_node = self.visit(value)
    #    return BroadcastTensor(dbg, value_node, factor)

    def visit_BinOp(self, binOp: ast.BinOp):
        if isinstance(binOp.op, ast.Add):
            return self.visit_BinOp_Add(binOp)
        if isinstance(binOp.op, ast.Sub):
            return self.visit_BinOp_Sub(binOp)
        elif isinstance(binOp.op, ast.BitOr):
            return self.visit_BinOp_BitOr(binOp)
        elif isinstance(binOp.op, ast.Mult):
            return self.visit_BinOp_Mult(binOp)
        elif isinstance(binOp.op, ast.RShift):
            return self.visit_BinOp_RShift(binOp)
        #elif isinstance(binOp.op, ast.BitAnd):
        #    return self.visit_BinOp_BitAnd(binOp)
        #elif isinstance(binOp.op, ast.MatMult):
        #    return self.visit_BinOp_MatMult(binOp)
        else:
            op_name = type(binOp.op).__name__
            raise QwertySyntaxError('Unknown binary operation {}'
                                    .format(op_name),
                                    self.get_debug_loc(binOp))

    def visit_BinOp_Add(self, binOp: ast.BinOp):
        # A top-level `+` expression must be a qubit literal (a superpos)
        qlit = self.extract_qubit_literal(binOp)
        return QpuExpr.new_qlit(qlit)

    def visit_BinOp_Sub(self, binOp: ast.BinOp):
        # A top-level `-` expression must be a qubit literal (a superpos with a
        # negative phase)
        qlit = self.extract_qubit_literal(binOp)
        return QpuExpr.new_qlit(qlit)

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

    def visit_BinOp_Mult(self, binOp: ast.BinOp):
        left = self.visit(binOp.left)
        right = self.visit(binOp.right)
        dbg = self.get_debug_loc(binOp)
        return QpuExpr.new_tensor([left, right], dbg)

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

    #def list_comp_helper(self, gen: Union[ast.GeneratorExp, ast.ListComp]):
    #    """
    #    Helper used by both ``visit_GeneratorExp()`` and ``visit_ListComp()``,
    #    since both Python AST nodes have near-identical fields.
    #    """
    #    dbg = self.get_debug_loc(gen)

    #    if len(gen.generators) != 1:
    #        raise QwertySyntaxError('Multiple generators are unsupported in '
    #                                'Qwerty', self.get_debug_loc(gen))
    #    comp = gen.generators[0]

    #    if comp.ifs:
    #        raise QwertySyntaxError('"if" not supported inside repeat '
    #                                'construct',
    #                                self.get_debug_loc(gen))
    #    if comp.is_async:
    #        raise QwertySyntaxError('async generators not supported',
    #                                self.get_debug_loc(gen))

    #    if not isinstance(comp.target, ast.Name) \
    #            or not isinstance(comp.iter, ast.Call) \
    #            or not isinstance(comp.iter.func, ast.Name) \
    #            or comp.iter.func.id != 'range' \
    #            or len(comp.iter.args) != 1 \
    #            or comp.iter.keywords:
    #        raise QwertySyntaxError('Unsupported generator syntax (only '
    #                                'basic "x for i in range(N) is '
    #                                'supported")',
    #                                self.get_debug_loc(gen))

    #    loopvar = comp.target.id

    #    if loopvar in self.dim_vars:
    #        raise QwertySyntaxError('Index variable {} collides with the '
    #                                'name of another type variable'
    #                                .format(loopvar),
    #                                self.get_debug_loc(gen))
    #    self.dim_vars.add(loopvar)
    #    body = self.visit(gen.elt)
    #    self.dim_vars.remove(loopvar)

    #    ub = self.extract_dimvar_expr(comp.iter.args[0])

    #    return (dbg, body, loopvar, ub)

    #def visit_GeneratorExp(self, gen: ast.GeneratorExp):
    #    """
    #    Convert a Python generator expression AST node into a Qwerty ``Repeat``
    #    AST node. For example the highlighted part of the code::

    #        ... | (func for i in range(20)) | ...
    #               ^^^^^^^^^^^^^^^^^^^^^^^

    #    is converted to a Repeat AST node. Here, ``range`` is a keyword whose
    #    operand is a dimension variable expression.
    #    """
    #    return Repeat(*self.list_comp_helper(gen))

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
                name_node, = args
                if not isinstance(name_node, ast.Name):
                    node_name = type(name_node).__name__
                    raise QwertySyntaxError('Classical embedding intrinsics '
                                            'operate on only variable names, '
                                            'such as `my_func.sign`, but '
                                            f'found {node_name} syntax '
                                            'instead.', dbg)

                func_name = name_node.id
                try:
                    captured = self.capturer.capture(func_name)
                except CaptureError:
                    raise QwertySyntaxError('Cannot create quantum embedding '
                                            f'of Python object {func_name} '
                                            'because it is not a @classical '
                                            'function.', dbg)
                if captured is None:
                    raise QwertySyntaxError('There is no classical function '
                                            f'named {func_name} to embed.',
                                            dbg)
                elif not isinstance(captured, CapturedSymbol):
                    raise QwertySyntaxError(f'The variable {func_name} is not '
                                            'a classical function.', dbg)

                mangled_name = captured.mangled_name
                embed_kind = EMBED_KINDS[intrinsic_name]
                return QpuExpr.new_embed_classical(mangled_name, embed_kind, dbg)
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

    #def visit_Attribute(self, attr: ast.Attribute):
    #    """
    #    Convert a Python attribute access AST node into Qwerty AST nodes for
    #    primitives such as ``.measure`` or ``.flip``. For example,
    #    ``std.measure`` becomes a ``Measure`` AST node with a ``BuiltinBasis``
    #    child node.
    #    """
    #    dbg = self.get_debug_loc(attr)
    #    attr_lhs = attr.value
    #    attr_rhs = attr.attr

    #    if attr_rhs == 'measure':
    #        basis = self.extract_basis(attr_lhs)
    #        return Expr.new_measure(basis, dbg)
    #    elif attr_rhs == 'project':
    #        basis = self.visit(attr_lhs)
    #        return Project(dbg, basis)
    #    elif attr_rhs == 'q':
    #        bits = self.visit(attr_lhs)
    #        return Lift(dbg, bits)
    #    elif attr_rhs == 'prep':
    #        operand = self.visit(attr_lhs)
    #        return Prepare(dbg, operand)
    #    elif attr_rhs == 'flip':
    #        operand = self.visit(attr_lhs)
    #        return Flip(dbg, operand)
    #    elif attr_rhs in EMBEDDING_KEYWORDS:
    #        if isinstance(attr_lhs, ast.Name):
    #            name = attr_lhs
    #            classical_func_name = name.id
    #        else:
    #            raise QwertySyntaxError('Keyword {} must be applied to an '
    #                                    'identifier, not a {}'
    #                                    .format(attr_rhs,
    #                                            type(attr_lhs).__name__),
    #                                    self.get_debug_loc(attr))

    #        embedding_kind = EMBEDDING_KEYWORDS[attr_rhs]

    #        if embedding_kind_has_operand(embedding_kind):
    #            raise QwertySyntaxError('Keyword {} requires an operand, '
    #                                    'specified with .{}(...)'
    #                                    .format(attr_rhs, attr_rhs),
    #                                    self.get_debug_loc(attr))

    #        return EmbedClassical(dbg, classical_func_name, '', embedding_kind)
    #    else:
    #        raise QwertySyntaxError('Unsupported keyword {}'.format(attr_rhs),
    #                                self.get_debug_loc(attr))

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
            pred_basis = self.extract_basis(if_expr.test)
        # TODO: do a more granular catch here
        except QwertySyntaxError:
            cond_expr = self.visit(if_expr.test)
            return QpuExpr.new_conditional(then_expr, else_expr, cond_expr, dbg)
        else:
            return QpuExpr.new_predicated(then_expr, else_expr, pred_basis, dbg)

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
        #elif isinstance(node, ast.Subscript):
        #    return self.visit_Subscript(node)
        #elif isinstance(node, ast.BinOp):
        elif isinstance(node, ast.BinOp):
            return self.visit_BinOp(node)
        #elif isinstance(node, ast.UnaryOp):
        #    return self.visit_UnaryOp(node)
        elif isinstance(node, ast.Set):
            return self.visit_Set(node)
        #elif isinstance(node, ast.GeneratorExp):
        #    return self.visit_GeneratorExp(node)
        #elif isinstance(node, ast.ListComp):
        #    return self.visit_ListComp(node)
        elif isinstance(node, ast.List):
            return self.visit_List(node)
        elif isinstance(node, ast.Call):
            return self.visit_Call(node)
        #elif isinstance(node, ast.Tuple):
        #    return self.visit_Tuple(node)
        #elif isinstance(node, ast.Attribute):
        #    return self.visit_Attribute(node)
        elif isinstance(node, ast.IfExp):
            return self.visit_IfExp(node)
        #elif isinstance(node, ast.BoolOp):
        #    return self.visit_BoolOp(node)
        else:
            return self.base_visit(node)

def convert_qpu_ast(module: ast.Module, name_generator: Callable[[str], str],
                    capturer: Capturer, filename: str = '',
                    line_offset: int = 0, col_offset: int = 0) -> QpuFunctionDef:
    """
    Run the ``QpuVisitor`` on the provided Python AST to convert to a Qwerty
    ``@qpu`` AST and return the result. The return value is the same as
    ``convert_ast()`` above.
    """
    if not isinstance(module, ast.Module):
        raise QwertySyntaxError('Expected top-level Module node in Python AST',
                                None) # This should not happen

    visitor = QpuVisitor(name_generator, capturer, filename, line_offset,
                         col_offset)
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
    visitor = QpuVisitor(name_generator=lambda name: name,
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
        #elif isinstance(binOp.op, ast.Mod):
        #    return self.visit_BinOp_Mod(binOp)
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
#
#    # Modular multiplication: X**2**J*y % N
#    def visit_BinOp_Mod(self, mod: ast.BinOp):
#        """
#        Convert a Python expression ``X**2**J*y % N`` to a Qwerty ``ModMulOp``
#        AST node. This is useful for the order finding oracle in the order
#        finding subroutine of Shor's algorithm.
#        """
#        if not isinstance((left_mul := mod.left), ast.BinOp) \
#                or not isinstance(left_mul.op, ast.Mult) \
#                or not isinstance((pow_ := left_mul.left), ast.BinOp) \
#                or not isinstance(pow_.op, ast.Pow) \
#                or not isinstance((inner_pow := pow_.right), ast.BinOp) \
#                or not isinstance(inner_pow.op, ast.Pow):
#            raise QwertySyntaxError('Unknown modulo syntax',
#                                    self.get_debug_loc(mod))
#
#        x = self.extract_dimvar_expr(pow_.left)
#        exp_base = self.extract_dimvar_expr(inner_pow.left)
#        j = self.extract_dimvar_expr(inner_pow.right)
#        y = self.visit(left_mul.right)
#        modN = self.extract_dimvar_expr(mod.right)
#
#        if not exp_base.is_constant():
#            raise QwertySyntaxError('Dimvars not allowed in base of exponent in '
#                                    'modular multiplication',
#                                    self.get_debug_loc(inner_pow.left))
#        if exp_base.get_value() != 2:
#            raise QwertySyntaxError('Currently, only 2 is supported as the '
#                                    'base of the exponent in modular '
#                                    'multiplication',
#                                    self.get_debug_loc(inner_pow.left))
#
#        dbg = self.get_debug_loc(mod)
#        return ModMulOp(dbg, x, j, y, modN)

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
#
#    def visit_Subscript(self, sub: ast.Subscript):
#        """
#        Convert a Python getitem expression to Qwerty ``Slice`` AST node.
#        """
#        val = self.visit(sub.value)
#        if isinstance(sub.slice, ast.Slice):
#            if sub.slice.step is not None:
#                raise QwertySyntaxError('[:::] syntax not supported',
#                                        self.get_debug_loc(sub))
#            if sub.slice.lower is None:
#                lower = None
#            else:
#                lower = self.extract_dimvar_expr(sub.slice.lower)
#
#            if sub.slice.upper is None:
#                upper = None
#            else:
#                upper = self.extract_dimvar_expr(sub.slice.upper)
#        else:
#            lower = self.extract_dimvar_expr(sub.slice)
#            upper = lower.copy()
#            upper += DimVarExpr("", 1)
#        dbg = self.get_debug_loc(sub)
#        return Slice(dbg, val, lower, upper)
#
    def visit(self, node: ast.AST):
        if isinstance(node, ast.UnaryOp):
            return self.visit_UnaryOp(node)
        elif isinstance(node, ast.BinOp):
            return self.visit_BinOp(node)
        elif isinstance(node, ast.Call):
            return self.visit_Call(node)
        #elif isinstance(node, ast.Tuple):
        #    return self.visit_Tuple(node)
        #elif isinstance(node, ast.Subscript):
        #    return self.visit_Subscript(node)
        else:
            return self.base_visit(node)

def convert_classical_ast(module: ast.Module,
                          name_generator: Callable[[str], str],
                          capturer: Capturer, filename: str = '',
                          line_offset: int = 0, col_offset: int = 0) \
                         -> ClassicalFunctionDef:
    """
    Run the ``ClassicalVisitor`` on the provided Python AST to convert to a
    Qwerty ``@classical`` AST and return the result. The return value is the
    same as ``convert_ast()`` above.
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
