"""
Convert a Python AST to a Qwerty AST by recognizing patterns in the Python AST
formed by Qwerty syntax.
"""

import ast
import math
from collections.abc import Callable
from enum import Enum
from typing import List, Tuple, Union
from .err import EXCLUDE_ME_FROM_STACK_TRACE_PLEASE, QwertySyntaxError, \
                 _get_frame
from ._qwerty_pyrt import DebugLoc, RegKind, Type, FunctionDef

#################### COMMON CODE FOR BOTH @QPU AND @CLASSICAL DSLs ####################

class AstKind(Enum):
    QPU = 1
    CLASSICAL = 2

def convert_ast(ast_kind: AstKind, module: ast.Module,
                name_generator: Callable[[str], str], filename: str = '',
                line_offset: int = 0, col_offset: int = 0) -> FunctionDef:
    """
    Take in a Python AST for a function parsed with ``ast.parse(mode='exec')``
    and return a ``Kernel`` Qwerty AST node.

    The ``line_offset`` and `col_offset`` are useful (respectively) because a
    ``@qpu``/``@classical`` kernel may begin after the first line of the file,
    and the caller may de-indent source code to avoid angering ``ast.parse()``.
    """
    if ast_kind == AstKind.QPU:
        return convert_qpu_ast(module, name_generator, filename, line_offset,
                               col_offset)
    #elif ast_kind == AstKind.CLASSICAL:
    #    return convert_classical_ast(module, filename, line_offset, col_offset)
    else:
        raise ValueError('unknown AST type {}'.format(ast_kind))

class BaseVisitor:
    """
    Common Python AST visitor for both ``@classical`` and ``@qpu`` kernels.
    """

    def __init__(self, name_generator: Callable[[str], str],
                 filename: str = '', line_offset: int = 0, col_offset: int = 0,
                 no_pyframe: bool = False):
        """
        Constructor. The ``no_pyframe`` flag is used by the tests to avoid
        including frames (see ``errs.py``) in DebugInfos constructed by
        ``get_debug_info()`` below, since this complicates testing.
        """
        self.name_generator = name_generator
        self.filename = filename
        self.line_offset = line_offset
        self.col_offset = col_offset
        self.frame = None if no_pyframe else _get_frame()

    def get_node_row_col(self, node: ast.AST):
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            row = node.lineno + self.line_offset
            col = node.col_offset + 1 + self.col_offset
            return row, col
        else:
            return None, None

    def get_debug_info(self, node: ast.AST) -> DebugLoc:
        """
        Extract line and column number from a Python AST node and return a
        Qwerty DebugInfo instance.
        """
        row, col = self.get_node_row_col(node)
        #return DebugLoc(self.filename, row or 0, col or 0, self.frame)
        return DebugLoc(self.filename, row or 0, col or 0)

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
    #                                    self.get_debug_info(node))
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
    #                                    self.get_debug_info(node))
    #        left *= right
    #        return left
    #    else:
    #        raise QwertySyntaxError('Unsupported constant type expression',
    #                                self.get_debug_info(node))

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
        #func_type_aliases = {'qfunc': (Type.new_qubit, Type.new_func),
        #                     'cfunc': (Type.new_bit, Type.new_func),
        #                     'rev_qfunc': (Type.new_qubit, Type.new_rev_func)}

        if isinstance(node, ast.Name):
            type_name = node.id
            if type_name == 'qubit':
                return Type.new_reg(RegKind.Qubit, 1)
            elif type_name == 'bit':
                return Type.new_reg(RegKind.Bit, 1)
            #elif type_name == 'int':
            #    return Type.new_int()
            #elif type_name == 'angle':
            #    return Type.new_float()
            #elif type_name == 'ampl':
            #    return Type.new_complex()
            elif type_name == 'qfunc':
                return Type.new_func(Type.new_reg(RegKind.Qubit, 1),
                                     Type.new_reg(RegKind.Qubit, 1))
            elif type_name == 'rev_qfunc':
                return Type.new_rev_func(Type.new_reg(RegKind.Qubit, 1))
            #elif type_name in func_type_aliases:
            #    new_val, new_func = func_type_aliases[type_name]
            #    return new_func(new_val(DimVarExpr('', 1)),
            #                    new_val(DimVarExpr('', 1)))
            else:
                raise QwertySyntaxError('Unknown type name {} found'
                                        .format(type_name),
                                        self.get_debug_info(node))
        #elif isinstance(node, ast.Subscript) \
        #        and isinstance(node.value, ast.Name) \
        #        and (node.value.id in func_type_aliases
        #             or node.value.id == 'func'
        #             or node.value.id == 'rev_func'):
        #    type_name = node.value.id
        #    if type_name in func_type_aliases:
        #        dims = self.extract_comma_sep_dimvar_expr(node.slice)
        #        new_val, new_func = func_type_aliases[type_name]
        #        in_type = new_val(dims[0])
        #        if len(dims) == 1:
        #            out_type = in_type.copy()
        #        elif len(dims) == 2:
        #            out_type = new_val(dims[1])
        #        else:
        #            raise QwertySyntaxError('Unsupported number of {} args '
        #                                    '{}. Expected 1 or 2'
        #                                    .format(type_name, len(dims)),
        #                                    self.get_debug_info(node))
        #        return new_func(in_type, out_type)
        #    elif type_name in ('func', 'rev_func'):
        #        if not isinstance(node.slice, ast.Tuple) \
        #                or len(node.slice.elts) != 2 \
        #                or not isinstance(node.slice.elts[0], ast.List):
        #            raise QwertySyntaxError('Invalid form of func[[arg1,arg2,...,argn], ret] '
        #                                    'found', self.get_debug_info(node))
        #        args_list, ret = node.slice.elts
        #        args = args_list.elts
        #        if len(args) == 0:
        #            arg_type = Type.new_tuple()
        #        elif len(args) == 1:
        #            arg_type = self.extract_type_literal(args[0])
        #        else:
        #            arg_type = Type.new_tuple([self.extract_type_literal(arg) for arg in args])
        #        ret_type = self.extract_type_literal(ret)
        #        if type_name == 'rev_func':
        #            new_func = Type.new_rev_func
        #        else:
        #            new_func = Type.new_func
        #        return new_func(arg_type, ret_type)
        #    else:
        #        raise QwertySyntaxError('Unknown alias {}[...] found'
        #                                .format(type_name),
        #                                self.get_debug_info(node))
        #elif isinstance(node, ast.Subscript) and node.value :
        #    elem_type = self.extract_type_literal(node.value)
        #    factor = self.extract_dimvar_expr(node.slice)
        #    return Type.new_broadcast(elem_type, factor)
        else:
            raise QwertySyntaxError('Unknown type',
                                    self.get_debug_info(node))

    def visit_Module(self, module: ast.Module):
        """
        Root node of Python AST
        """
        # No idea what this is, so conservatively reject it
        if module.type_ignores:
            raise QwertySyntaxError('I do not understand type_ignores, but '
                                    'they were specified in a Python module',
                                    self.get_debug_info(module))

        if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
            raise QwertySyntaxError('Expected exactly 1 FunctionDef in '
                                    'module body',
                                    self.get_debug_info(module))

        func_def = module.body[0]
        return self.visit_FunctionDef(func_def)

    # Root note for a single expression
    #def visit_Expression(self, expr: ast.Expression):
    #    return self.visit(expr.body)

    def base_visit_FunctionDef(self, func_def: ast.FunctionDef,
                               decorator_name: str) -> FunctionDef:
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
                                        self.get_debug_info(func_def))

        is_rev_dec = lambda dec: isinstance(dec, ast.Name) and dec.id == 'reversible'
        if (n_decorators := len(func_def.decorator_list)) > 2:
            raise QwertySyntaxError('Wrong number of decorators ({} > 2) '
                                    'for {}()'.format(n_decorators,
                                                      func_name),
                                    self.get_debug_info(func_def))
        elif n_decorators == 2:
            rev_decorator, our_decorator = func_def.decorator_list
            if not is_rev_dec(rev_decorator):
                # swap
                our_decorator, rev_decorator = rev_decorator, our_decorator
            if not is_rev_dec(rev_decorator):
                raise QwertySyntaxError('Unknown decorator {} on {}()'
                                        .format(rev_decorator,
                                                func_name),
                                        self.get_debug_info(func_def))
            # By this point, one of the decorators is @reversible
            is_rev = True
        elif n_decorators == 1:
            our_decorator, = func_def.decorator_list
            is_rev = False
        else: # n_decorators == 0
            raise QwertySyntaxError('No decorators (e.g., @{}) for {}()'
                                    .format(decorator_name, func_name),
                                    self.get_debug_info(func_def))

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
                                    self.get_debug_info(our_decorator))

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
                                        self.get_debug_info(arg))
            if not arg_type:
                raise QwertySyntaxError('Currently, type annotations are '
                                        'required for arguments such as {} '
                                        'argument of {}()'
                                        .format(arg_name, func_name),
                                        self.get_debug_info(arg))

            actual_arg_type = self.extract_type_literal(arg_type)
            args.append((actual_arg_type, arg_name))

        # Now, figure out return type
        if not func_def.returns:
            raise QwertySyntaxError('Currently, type annotations are '
                                    'required for functions such as {}() '
                                    .format(func_name),
                                    self.get_debug_info(func_def))
        ret_type = self.extract_type_literal(func_def.returns)

        # Great, now we have everything we need to build the AST node...
        dbg = self.get_debug_info(func_def)

        # ...except traversing the function body
        #body = self.visit(func_def.body)
        #return Kernel(dbg, ast_kind, func_name, func_type, capture_names,
        #              capture_types, capture_freevars, arg_names, dim_vars,
        #              body)

        generated_func_name = self.name_generator(func_name)
        return FunctionDef(generated_func_name, args, ret_type, dbg)

    #def visit_List(self, nodes: List[ast.AST]):
    #    """
    #    Convenience function to visit each node in a ``list`` and return the
    #    results of each as a new list.
    #    """
    #    return [self.visit(node) for node in nodes]

    #def visit_Return(self, ret: ast.Return):
    #    """
    #    Convert a Python ``return`` statement into a Qwerty AST ``Return``
    #    node.
    #    """
    #    dbg = self.get_debug_info(ret)
    #    expr = self.visit(ret.value)
    #    return Return(dbg, expr)

    #def visit_Assign(self, assign: ast.Assign) -> Assign:
    #    """
    #    Convert a Python assignment statement::

    #        q = '0'

    #    into an ``Assign`` Qwerty AST node.

    #    A destructuring assignment::

    #        q1, q2 = '01'

    #    is converted into a ``DestructAssign`` Qwerty AST node.
    #    """
    #    dbg = self.get_debug_info(assign)
    #    if len(assign.targets) != 1:
    #        # Something like a = b = c = '0'
    #        raise QwertySyntaxError("Multiple assignments (like a = b = '0') "
    #                                "are not supported. Please write a = '0' "
    #                                "and then b = '0' instead", dbg)
    #    tgt, = assign.targets

    #    if isinstance(tgt, ast.Name):
    #        expr = self.visit(assign.value)
    #        name = tgt.id

    #        if name in RESERVED_KEYWORDS:
    #            raise QwertySyntaxError('{} is a reserved keyword and cannot '
    #                                    'be used as the left-hand side of an '
    #                                    'assignment'.format(name), dbg)

    #        return Assign(dbg, name, expr)
    #    elif isinstance(tgt, ast.Tuple) \
    #            and all(isinstance(elt, ast.Name) for elt in tgt.elts):
    #        names = [name.id for name in tgt.elts]

    #        if len(names) < 2:
    #            raise QwertySyntaxError('Destructuring assignment must have '
    #                                    'at least 2 names on the left-hand '
    #                                    'side', dbg)

    #        if bad_names := RESERVED_KEYWORDS & set(names):
    #            bad_name = next(iter(bad_names))
    #            raise QwertySyntaxError('{} is a reserved keyword and cannot '
    #                                    'be used as the left-hand side of an '
    #                                    'assignment'.format(bad_name), dbg)

    #        expr = self.visit(assign.value)
    #        return DestructAssign(dbg, names, expr)
    #    else:
    #        raise QwertySyntaxError('Unknown assignment syntax', dbg)

    #def visit_AnnAssign(self, assign: ast.AnnAssign):
    #    """
    #    Throw an error for the Python type-annotated assignment statement,
    #    since it is unnecessary::

    #        q: qubit = '0'
    #    """
    #    dbg = self.get_debug_info(assign)
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

    def base_visit(self, node: ast.AST):
        """
        Convert a Python AST node into a Qwerty AST Node (and return the
        latter).
        """
        if isinstance(node, list):
            return self.visit_List(node)
        elif isinstance(node, ast.Return):
            return self.visit_Return(node)
        elif isinstance(node, ast.Assign):
            return self.visit_Assign(node)
        elif isinstance(node, ast.AnnAssign):
            return self.visit_AnnAssign(node)
        elif isinstance(node, ast.AugAssign):
            return self.visit_AugAssign(node)
        # Commenting these for now, since we can't handle nested functions, and
        # a nested module doesn't make much sense
        #elif isinstance(node, ast.Module):
        #    return self.visit_Module(node)
        #elif isinstance(node, ast.FunctionDef):
        #    return self.visit_FunctionDef(node)
        else:
            node_name = type(node).__name__
            raise QwertySyntaxError(f'Unknown Python AST node {node_name}',
                                    self.get_debug_info(node))

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

class QpuVisitor(BaseVisitor):
    """
    Python AST visitor for syntax specific to ``@qpu`` kernels.
    """

    def __init__(self, name_generator: Callable[[str], str],
                 filename: str = '', line_offset: int = 0,
                 col_offset: int = 0, no_pyframe: bool = False):
        super().__init__(name_generator, filename, line_offset, col_offset,
                         no_pyframe)

    #def extract_float_expr(self, node: ast.AST):
    #    """
    #    Extract a float expression, like a tilt, for example::

    #        '1' @ (3*pi/2)
    #               ^^^^^^
    #    """
    #    dbg = self.get_debug_info(node)

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
    #                                self.get_debug_info(node))

    def visit_FunctionDef(self, func_def: ast.FunctionDef) -> FunctionDef:
        """
        Convert a ``@qpu`` kernel into a ``QpuKernel`` Qwerty AST node.
        """
        return super().base_visit_FunctionDef(func_def, 'qpu')

    ## Variable or reserved keyword
    #def visit_Name(self, name: ast.Name):
    #    """
    #    Convert a Python AST identitifer node into either a Qwerty primitive
    #    AST node or a Qwerty variable name AST node. For example ``id`` becomes
    #    an ``Identity`` AST node, and ``foobar`` becomes a Qwerty ``Variable``
    #    AST node.
    #    """
    #    var_name = name.id
    #    dbg = self.get_debug_info(name)

    #    if var_name == 'id':
    #        return Identity(dbg, DimVarExpr('', 1))
    #    elif var_name == 'std':
    #        return BuiltinBasis(dbg, Z, DimVarExpr('', 1))
    #    elif var_name == 'pm':
    #        return BuiltinBasis(dbg, X, DimVarExpr('', 1))
    #    elif var_name == 'ij':
    #        return BuiltinBasis(dbg, Y, DimVarExpr('', 1))
    #    elif var_name == 'discard':
    #        return Discard(dbg, DimVarExpr('', 1))
    #    elif var_name == 'measure':
    #        # Sugar for `std.measure'
    #        return Measure(dbg, BuiltinBasis(dbg.copy(), Z, DimVarExpr('', 1)))
    #    elif var_name == 'flip':
    #        # Sugar for `std.flip'
    #        return Flip(dbg, BuiltinBasis(dbg.copy(), Z, DimVarExpr('', 1)))
    #    elif var_name == 'fourier':
    #        raise QwertySyntaxError('fourier is a reserved keyword. The '
    #                                'one-dimensional Fourier basis must be '
    #                                'written as fourier[1]',
    #                                self.get_debug_info(name))
    #    else:
    #        return Variable(dbg, var_name)

    #def visit_Constant(self, const: ast.Constant):
    #    """
    #    Convert a Python string literal into a Qwerty ``QubitLiteral`` AST
    #    node. Since the ``QubitLiteral`` Qwerty AST node supports only one
    #    eigenstate and primitive basis, this produces a tree of ``BiTensor``s
    #    concatenating multiple ``QubitLiteral`` nodes as necessary.
    #    For example, ``'0011'`` becomes a ``BiTensor`` of two different
    #    ``QubitLiteral`` nodes.
    #    """
    #    value = const.value
    #    if isinstance(value, str):
    #        state_chars = value
    #        if not state_chars:
    #            raise QwertySyntaxError('Qubit literal must not be an empty string',
    #                                    self.get_debug_info(const))

    #        result = None
    #        last_char, last_dim = None, None

    #        def add_to_tensor_tree():
    #            nonlocal result, last_char, last_dim

    #            dbg = self.get_debug_info(const)
    #            eigenstate, prim_basis = STATE_CHAR_MAPPING[last_char]
    #            vec = QubitLiteral(dbg, eigenstate, prim_basis, last_dim)

    #            if result is None:
    #                result = vec
    #            else:
    #                # Need to make a second DebugInfo since the last one is going
    #                # to get std::move()'d away by the QubitLiteral constructor above
    #                dbg = self.get_debug_info(const)
    #                result = BiTensor(dbg, result, vec)

    #        for i, c in enumerate(state_chars):
    #            if c not in STATE_CHAR_MAPPING:
    #                raise QwertySyntaxError('Unknown state |{}⟩ in qubit literal'
    #                                        .format(c),
    #                                        self.get_debug_info(const))
    #            if last_char == c:
    #                last_dim += DimVarExpr('', 1)
    #            else:
    #                if last_char is not None:
    #                    add_to_tensor_tree()
    #                last_char = c
    #                last_dim = DimVarExpr('', 1)

    #            if i+1 == len(state_chars):
    #                add_to_tensor_tree()

    #        return result
    #    else:
    #        raise QwertySyntaxError('Unknown constant syntax',
    #                                self.get_debug_info(const))

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
    #    dbg = self.get_debug_info(subscript)

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

    #def visit_BinOp(self, binOp: ast.BinOp):
    #    if isinstance(binOp.op, ast.Add):
    #        return self.visit_BinOp_Add(binOp)
    #    elif isinstance(binOp.op, ast.BitOr):
    #        return self.visit_BinOp_BitOr(binOp)
    #    elif isinstance(binOp.op, ast.RShift):
    #        return self.visit_BinOp_RShift(binOp)
    #    elif isinstance(binOp.op, ast.BitAnd):
    #        return self.visit_BinOp_BitAnd(binOp)
    #    elif isinstance(binOp.op, ast.MatMult):
    #        return self.visit_BinOp_MatMult(binOp)
    #    else:
    #        op_name = type(binOp.op).__name__
    #        raise QwertySyntaxError('Unknown binary operation {}'
    #                                .format(op_name),
    #                                self.get_debug_info(binOp))

    #def visit_BinOp_Add(self, binOp: ast.BinOp):
    #    """
    #    Convert a Python binary add expression into a Qwerty tensor product AST
    #    node. For example, ``t1 + t2`` becomes a ``BiTensor`` Qwerty AST node
    #    with two children.
    #    """
    #    left = self.visit(binOp.left)
    #    right = self.visit(binOp.right)
    #    dbg = self.get_debug_info(binOp)
    #    return BiTensor(dbg, left, right)

    #def visit_BinOp_BitOr(self, binOp: ast.BinOp):
    #    """
    #    Convert a Python bitwise OR expression into a Qwerty ``Pipe`` (function
    #    call) AST node. For example, ``t1 | t2`` becomes a ``Pipe`` node with
    #    two children.
    #    """
    #    left = self.visit(binOp.left)
    #    right = self.visit(binOp.right)
    #    dbg = self.get_debug_info(binOp)
    #    return Pipe(dbg, left, right)

    #def visit_BinOp_BitAnd(self, binOp: ast.BinOp):
    #    """
    #    Convert a Python bitwise AND expression into a Qwerty ``Pred``
    #    (predication) AST node. For example, ``t1 & t2`` becomes a ``Pred``
    #    node with two children — one should be a basis and one should be a
    #    function.
    #    """
    #    basis = self.visit(binOp.left)
    #    body = self.visit(binOp.right)
    #    dbg = self.get_debug_info(binOp)
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
    #                self.get_debug_info(binOp))
    #        if len(call.args) != 1:
    #            raise QwertySyntaxError(
    #                'Wrong number of arguments {} != 1 passed to {}(...)'
    #                .format(len(call.args), unit),
    #                self.get_debug_info(binOp))
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
    #        angle_conv_dbg = self.get_debug_info(angle_conv_dbg_node)
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

    #    dbg = self.get_debug_info(binOp)
    #    lhs = self.visit(binOp.left)
    #    return Phase(dbg, angle_expr, lhs)

    #def visit_BinOp_RShift(self, binOp: ast.BinOp):
    #    """
    #    Convert a Python right bit shift AST node to a Qwerty
    #    ``BasisTranslation`` AST node. For example, ``b1 >> b2`` becomes a
    #    ``BasisTranslation`` AST node with two basis children.
    #    """
    #    basis_in, basis_out = binOp.left, binOp.right

    #    dbg = self.get_debug_info(binOp)
    #    basis_in_node = self.visit(basis_in)
    #    basis_out_node = self.visit(basis_out)
    #    return BasisTranslation(dbg, basis_in_node, basis_out_node)

    #def visit_UnaryOp(self, unaryOp: ast.UnaryOp):
    #    if isinstance(unaryOp.op, ast.USub):
    #        return self.visit_UnaryOp_USub(unaryOp)
    #    elif isinstance(unaryOp.op, ast.Invert):
    #        return self.visit_UnaryOp_Invert(unaryOp)
    #    else:
    #        op_name = type(unaryOp.op).__name__
    #        raise QwertySyntaxError('Unknown unary operation {}'
    #                                .format(op_name),
    #                                self.get_debug_info(unaryOp))

    #def visit_UnaryOp_USub(self, unaryOp: ast.UnaryOp):
    #    """
    #    Convert a Python unary negation AST node into a Qwerty AST node tilting
    #    the operand by 180 degrees. For example, ``-f`` or ``-'0'``.
    #    """
    #    dbg = self.get_debug_info(unaryOp)
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
    #    dbg = self.get_debug_info(unaryOp)
    #    operand = self.visit(unary_operand)
    #    return Adjoint(dbg, operand)

    #def visit_Set(self, set_: ast.Set):
    #    """
    #    Convert a Python set literal AST node into a Qwerty ``BasisLiteral``
    #    AST node. For example, ``{'0', -'1'}`` is a ``BasisLiteral`` node with
    #    two children.
    #    """
    #    dbg = self.get_debug_info(set_)
    #    basis_elts = self.visit(set_.elts)
    #    return BasisLiteral(dbg, basis_elts)

    #def list_comp_helper(self, gen: Union[ast.GeneratorExp, ast.ListComp]):
    #    """
    #    Helper used by both ``visit_GeneratorExp()`` and ``visit_ListComp()``,
    #    since both Python AST nodes have near-identical fields.
    #    """
    #    dbg = self.get_debug_info(gen)

    #    if len(gen.generators) != 1:
    #        raise QwertySyntaxError('Multiple generators are unsupported in '
    #                                'Qwerty', self.get_debug_info(gen))
    #    comp = gen.generators[0]

    #    if comp.ifs:
    #        raise QwertySyntaxError('"if" not supported inside repeat '
    #                                'construct',
    #                                self.get_debug_info(gen))
    #    if comp.is_async:
    #        raise QwertySyntaxError('async generators not supported',
    #                                self.get_debug_info(gen))

    #    if not isinstance(comp.target, ast.Name) \
    #            or not isinstance(comp.iter, ast.Call) \
    #            or not isinstance(comp.iter.func, ast.Name) \
    #            or comp.iter.func.id != 'range' \
    #            or len(comp.iter.args) != 1 \
    #            or comp.iter.keywords:
    #        raise QwertySyntaxError('Unsupported generator syntax (only '
    #                                'basic "x for i in range(N) is '
    #                                'supported")',
    #                                self.get_debug_info(gen))

    #    loopvar = comp.target.id

    #    if loopvar in self.dim_vars:
    #        raise QwertySyntaxError('Index variable {} collides with the '
    #                                'name of another type variable'
    #                                .format(loopvar),
    #                                self.get_debug_info(gen))
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

    #def visit_Call(self, call: ast.Call):
    #    """
    #    As syntactic sugar, convert a Python call expression into a ``Pipe``
    #    Qwerty AST node. In general, the shorthand::

    #        f(arg1,arg2,...,argn)

    #    is equivalent to::

    #        (arg1,arg2,...,argn) | f

    #    There is also unrelated special handling for ``b.rotate(theta)`` and
    #    ``fwd.inplace(rev)``.
    #    """
    #    if call.keywords:
    #        raise QwertySyntaxError('Keyword arguments not supported in '
    #                                'call', self.get_debug_info(call))

    #    dbg = self.get_debug_info(call)

    #    # Handling for b.rotate(theta) and fwd.inplace(rev)
    #    if isinstance(attr := call.func, ast.Attribute):
    #        if attr.attr == 'rotate':
    #            if len(call.args) != 1:
    #                raise QwertySyntaxError('Wrong number of operands '
    #                                        '{} != 1 to .rotate'
    #                                        .format(len(call.args)),
    #                                        self.get_debug_info(attr))
    #            arg = call.args[0]
    #            basis = self.visit(attr.value)
    #            theta = self.extract_float_expr(arg)
    #            return Rotate(dbg, basis, theta)
    #        elif attr.attr in EMBEDDING_KEYWORDS \
    #                and isinstance(name := attr.value, ast.Name):
    #            if not call.args:
    #                embedding_kind = EMBEDDING_KEYWORDS[attr.attr]
    #                if embedding_kind_has_operand(embedding_kind):
    #                    raise QwertySyntaxError('Keyword {} requires an '
    #                                            'operand'.format(attr.attr),
    #                                            self.get_debug_info(attr))
    #                classical_func_name = name.id
    #                return EmbedClassical(dbg, classical_func_name, '',
    #                                      embedding_kind)
    #            else:
    #                embedding_kind = EMBEDDING_KEYWORDS[attr.attr]
    #                if not embedding_kind_has_operand(embedding_kind):
    #                    raise QwertySyntaxError('Keyword {} does not require an '
    #                                            'operand'.format(attr.attr),
    #                                            self.get_debug_info(attr))

    #                if len(call.args) != 1:
    #                    raise QwertySyntaxError('Wrong number of operands '
    #                                            '{} != 1 to {}'
    #                                            .format(len(call.args),
    #                                                    attr.attr),
    #                                            self.get_debug_info(attr))
    #                arg = call.args[0]
    #                if not isinstance(arg, ast.Name):
    #                    raise QwertySyntaxError('Argument to {} must be an '
    #                                            'identifier, not a {}'
    #                                            .format(attr.attr,
    #                                                    type(arg).__name__),
    #                                            self.get_debug_info(attr))

    #                classical_func_name = name.id
    #                classical_func_operand_name = arg.id
    #                return EmbedClassical(dbg, classical_func_name,
    #                                      classical_func_operand_name,
    #                                      embedding_kind)

    #    rhs = self.visit(call.func)
    #    lhs_elts = [self.visit(arg) for arg in call.args]
    #    lhs = TupleLiteral(dbg.copy(), lhs_elts) \
    #          if len(call.args) != 1 \
    #          else self.visit(call.args[0])
    #    return Pipe(dbg, lhs, rhs)

    #def visit_Tuple(self, tuple_: ast.Tuple):
    #    """
    #    Convert a Python tuple literal into a Qwerty tuple literal. Trust me,
    #    this one is thrilling.
    #    """
    #    dbg = self.get_debug_info(tuple_)
    #    elts = self.visit(tuple_.elts)
    #    return TupleLiteral(dbg, elts)

    #def visit_Attribute(self, attr: ast.Attribute):
    #    """
    #    Convert a Python attribute access AST node into Qwerty AST nodes for
    #    primitives such as ``.measure`` or ``.flip``. For example,
    #    ``std.measure`` becomes a ``Measure`` AST node with a ``BuiltinBasis``
    #    child node.
    #    """
    #    dbg = self.get_debug_info(attr)
    #    attr_lhs = attr.value
    #    attr_rhs = attr.attr

    #    if attr_rhs == 'measure':
    #        basis = self.visit(attr_lhs)
    #        return Measure(dbg, basis)
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
    #                                    self.get_debug_info(attr))

    #        embedding_kind = EMBEDDING_KEYWORDS[attr_rhs]

    #        if embedding_kind_has_operand(embedding_kind):
    #            raise QwertySyntaxError('Keyword {} requires an operand, '
    #                                    'specified with .{}(...)'
    #                                    .format(attr_rhs, attr_rhs),
    #                                    self.get_debug_info(attr))

    #        return EmbedClassical(dbg, classical_func_name, '', embedding_kind)
    #    else:
    #        raise QwertySyntaxError('Unsupported keyword {}'.format(attr_rhs),
    #                                self.get_debug_info(attr))

    #def visit_IfExp(self, ifExp: ast.IfExp):
    #    """
    #    Convert a Python conditional expression AST node into a Qwerty
    #    classical branching AST node. For example, ``x if y or z`` becomes a
    #    Qwerty ``Conditional`` AST node with three children.
    #    """
    #    if_expr, then_expr, else_expr = ifExp.test, ifExp.body, ifExp.orelse
    #    dbg = self.get_debug_info(ifExp)

    #    if_expr_node = self.visit(if_expr)
    #    then_expr_node = self.visit(then_expr)
    #    else_expr_node = self.visit(else_expr)

    #    return Conditional(dbg, if_expr_node, then_expr_node, else_expr_node)

    #def visit_BoolOp(self, boolOp: ast.BoolOp):
    #    if isinstance(boolOp.op, ast.Or):
    #        return self.visit_BoolOp_Or(boolOp)
    #    else:
    #        op_name = type(boolOp.op).__name__
    #        raise QwertySyntaxError('Unknown boolean operation {}'
    #                                .format(op_name),
    #                                self.get_debug_info(boolOp))

    #def visit_BoolOp_Or(self, boolOp: ast.BoolOp):
    #    """
    #    Convert a Python Boolean expression with an ``or`` into a Qwerty
    #    superposition AST node. For example, ``0.25*'0' or 0.75*'1'`` becomes a
    #    ``SuperpositionLiteral`` AST node with two children.
    #    """
    #    dbg = self.get_debug_info(boolOp)
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
    #            operand_dbg = self.get_debug_info(operand)
    #            raise QwertySyntaxError(
    #                'Either all operands of a superposition operator should '
    #                'have explicit probabilities, or none should have '
    #                'explicit probabilities', operand_dbg)

    #        had_prob = has_prob

    #        if has_prob:
    #            if not isinstance(prob_node, ast.Constant):
    #                prob_dbg = self.get_debug_info(prob_node)
    #                raise QwertySyntaxError(
    #                    'Currently, probabilities in a superposition literal '
    #                    'must be integer constants', prob_dbg)
    #            prob_const_node = prob_node
    #            prob_val = prob_const_node.value

    #            if not isinstance(prob_val, float) \
    #                    and not isinstance(prob_val, int):
    #                prob_dbg = self.get_debug_info(prob_node)
    #                raise QwertySyntaxError(
    #                    'Probabilities in a superposition literal must be '
    #                    'floats, not {}'.format(str(type(prob_val))), prob_dbg)
    #        else:
    #            prob_val = 1.0 / len(operands)

    #        pair = (prob_val, self.visit(vec_node))
    #        pairs.append(pair)

    #    return SuperposLiteral(dbg, pairs)

    def visit(self, node: ast.AST):
        #if isinstance(node, ast.Name):
        #    return self.visit_Name(node)
        #elif isinstance(node, ast.Constant):
        #    return self.visit_Constant(node)
        #elif isinstance(node, ast.Subscript):
        #    return self.visit_Subscript(node)
        #elif isinstance(node, ast.BinOp):
        #    return self.visit_BinOp(node)
        #elif isinstance(node, ast.UnaryOp):
        #    return self.visit_UnaryOp(node)
        #elif isinstance(node, ast.Set):
        #    return self.visit_Set(node)
        #elif isinstance(node, ast.GeneratorExp):
        #    return self.visit_GeneratorExp(node)
        #elif isinstance(node, ast.ListComp):
        #    return self.visit_ListComp(node)
        #elif isinstance(node, ast.Call):
        #    return self.visit_Call(node)
        #elif isinstance(node, ast.Tuple):
        #    return self.visit_Tuple(node)
        #elif isinstance(node, ast.Attribute):
        #    return self.visit_Attribute(node)
        #elif isinstance(node, ast.IfExp):
        #    return self.visit_IfExp(node)
        #elif isinstance(node, ast.BoolOp):
        #    return self.visit_BoolOp(node)
        #else:
        #    return self.base_visit(node)
        return self.base_visit(node)

def convert_qpu_ast(module: ast.Module, name_generator: Callable[[str], str],
                    filename: str = '', line_offset: int = 0,
                    col_offset: int = 0) -> FunctionDef:
    """
    Run the ``QpuVisitor`` on the provided Python AST to convert to a Qwerty
    ``@qpu`` AST and return the result. The return value is the same as
    ``convert_ast()`` above.
    """
    if not isinstance(module, ast.Module):
        raise QwertySyntaxError('Expected top-level Module node in Python AST',
                                None) # This should not happen

    visitor = QpuVisitor(name_generator, filename, line_offset, col_offset)
    return visitor.visit_Module(module)

#def convert_qpu_expr(expr: ast.Expression, filename: str = '',
#                     line_offset: int = 0, col_offset: int = 0,
#                     no_pyframe: bool = False) -> Expr:
#    """
#    Convert an expression from a @qpu kernel instead of the whole thing.
#    Currently used only in unit tests. Someday could be used in a REPL, for
#    example.
#    """
#    if not isinstance(expr, ast.Expression):
#        raise QwertySyntaxError('Expected top-level Expression node in '
#                                'Python AST', None) # This should not happen
#
#    visitor = QpuVisitor(filename, line_offset, col_offset, no_pyframe)
#    return visitor.visit_Expression(expr)

#################### @CLASSICAL DSL ####################

#class ClassicalVisitor(BaseVisitor):
#    """
#    Python AST visitor for syntax specific to ``@classical`` kernels.
#    """
#    def __init__(self, filename: str = '', line_offset: int = 0,
#                 col_offset: int = 0, no_pyframe: bool = False):
#        super().__init__(filename, line_offset, col_offset, no_pyframe)
#
#    def visit_FunctionDef(self, func_def: ast.FunctionDef) -> Kernel:
#        """
#        Convert a ``@classical`` kernel into a ``ClassicalKernel`` Qwerty AST
#        node.
#        """
#        return super().base_visit_FunctionDef(func_def, 'classical', AST_CLASSICAL)
#
#    def visit_Name(self, name: ast.Name):
#        """
#        Convert a Python AST identitifer node into a Qwerty variable name AST
#        node. (The ``@classical`` DSL does not have reserved keywords.)
#        """
#        var_name = name.id
#        dbg = self.get_debug_info(name)
#        return Variable(dbg, var_name)
#
#    def visit_UnaryOp(self, unaryOp: ast.UnaryOp):
#        if isinstance(unaryOp.op, ast.Invert):
#            return self.visit_UnaryOp_Invert(unaryOp)
#        else:
#            op_name = type(unaryOp.op).__name__
#            raise QwertySyntaxError('Unknown unary operation {}'
#                                    .format(op_name),
#                                    self.get_debug_info(unaryOp))
#
#    def visit_UnaryOp_Invert(self, unaryOp: ast.UnaryOp):
#        """
#        Convert a Python bitwise complement AST node into the same thing in the
#        Qwerty AST. For example, ``~x`` becomes a ``BitUnaryOp`` Qwerty AST
#        node.
#        """
#        operand = self.visit(unaryOp.operand)
#        dbg = self.get_debug_info(unaryOp)
#        return BitUnaryOp(dbg, BIT_NOT, operand)
#
#    def visit_BinOp(self, binOp: ast.BinOp):
#        if isinstance(binOp.op, ast.BitAnd):
#            return self.visit_BinOp_BitAnd(binOp)
#        elif isinstance(binOp.op, ast.BitXor):
#            return self.visit_BinOp_BitXor(binOp)
#        elif isinstance(binOp.op, ast.BitOr):
#            return self.visit_BinOp_BitOr(binOp)
#        elif isinstance(binOp.op, ast.Mod):
#            return self.visit_BinOp_Mod(binOp)
#        else:
#            op_name = type(binOp.op).__name__
#            raise QwertySyntaxError('Unknown binary operation {}'
#                                    .format(op_name),
#                                    self.get_debug_info(binOp))
#
#    def visit_BinOp_BitAnd(self, binOp: ast.BinOp):
#        """
#        Convert a Python binary bitwise AND expression into the same thing in
#        the Qwerty AST. For example, ``x & y`` becomes a ``BitBinaryOp`` Qwerty
#        AST node.
#        """
#        left = self.visit(binOp.left)
#        right = self.visit(binOp.right)
#        dbg = self.get_debug_info(binOp)
#        return BitBinaryOp(dbg, BIT_AND, left, right)
#
#    def visit_BinOp_BitXor(self, binOp: ast.BinOp):
#        """
#        Convert a Python binary bitwise XOR expression into the same thing in
#        the Qwerty AST. For example, ``x ^ y`` becomes a ``BitBinaryOp`` Qwerty
#        AST node.
#        """
#        left = self.visit(binOp.left)
#        right = self.visit(binOp.right)
#        dbg = self.get_debug_info(binOp)
#        return BitBinaryOp(dbg, BIT_XOR, left, right)
#
#    def visit_BinOp_BitOr(self, binOp: ast.BinOp):
#        """
#        Convert a Python binary bitwise OR expression into the same thing in
#        the Qwerty AST. For example, ``x | y`` becomes a ``BitBinaryOp`` Qwerty
#        AST node.
#        """
#        left = self.visit(binOp.left)
#        right = self.visit(binOp.right)
#        dbg = self.get_debug_info(binOp)
#        return BitBinaryOp(dbg, BIT_OR, left, right)
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
#                                    self.get_debug_info(mod))
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
#                                    self.get_debug_info(inner_pow.left))
#        if exp_base.get_value() != 2:
#            raise QwertySyntaxError('Currently, only 2 is supported as the '
#                                    'base of the exponent in modular '
#                                    'multiplication',
#                                    self.get_debug_info(inner_pow.left))
#
#        dbg = self.get_debug_info(mod)
#        return ModMulOp(dbg, x, j, y, modN)
#
#    def visit_Call(self, call: ast.Call):
#        """
#        Convert a Python call expression into either a ``BitLiteral`` Qwerty
#        AST node (for e.g. ``bit[4](0b1101)``) or other bitwise operations
#        expressed as (pseudo)functions in Python syntax.
#
#        For example, ``x.repeat(N)`` is converted to a ``BitRepeat`` Qwerty AST
#        node with one child ``x``; ``x.rotl(y)`` is converted to an appropriate
#        ``BitBinaryOp`` node with two children; and ``x.xor_reduce()`` is
#        converted to a ``BitReduceOp`` node with one child.
#        """
#        if isinstance(call.func, ast.Subscript) \
#                and isinstance(call.func.value, ast.Name) \
#                and call.func.value.id == 'bit':
#            if len(call.args) != 1:
#                raise QwertySyntaxError('bit() expects one positional '
#                                        'argument: the value',
#                                        self.get_debug_info(call))
#            if call.keywords:
#                raise QwertySyntaxError('bit() does not accept keyword '
#                                        'arguments',
#                                        self.get_debug_info(call))
#            val = self.extract_dimvar_expr(call.args[0])
#            n_bits = self.extract_dimvar_expr(call.func.slice)
#            dbg = self.get_debug_info(call)
#            return BitLiteral(dbg, val, n_bits)
#        else: # xor_reduce(), and_reduce(), etc
#            func = call.func
#            if not isinstance(func, ast.Attribute):
#                raise QwertySyntaxEbrror('I expect function calls to be of the '
#                                        'form expression.FUNC(), but this call '
#                                        'is not',
#                                        self.get_debug_info(call))
#            attr = func
#            operand = attr.value
#            func_name = attr.attr
#
#            reduce_pseudo_funcs = {'xor_reduce': BIT_XOR,
#                                   'and_reduce': BIT_AND}
#            binary_pseudo_funcs = {'rotr': BIT_ROTR,
#                                   'rotl': BIT_ROTL}
#            if func_name in reduce_pseudo_funcs:
#                if call.args or call.keywords:
#                    raise QwertySyntaxError('Arguments cannot be passed to the '
#                                            'function call',
#                                            self.get_debug_info(call))
#                dbg = self.get_debug_info(call)
#                return BitReduceOp(dbg, reduce_pseudo_funcs[func_name],
#                                   self.visit(operand))
#            elif func_name in binary_pseudo_funcs:
#                if call.keywords:
#                    raise QwertySyntaxError('Keywords arguments not '
#                                            'supported to {}()'
#                                            .format(func_name),
#                                            self.get_debug_info(call))
#                if len(call.args) != 1:
#                    raise QwertySyntaxError('{}() expects one positional '
#                                            'argument: the rotation amount'
#                                            .format(func_name),
#                                            self.get_debug_info(call))
#                val = self.visit(operand)
#                amount_node = call.args[0]
#                amount = self.visit(amount_node)
#                dbg = self.get_debug_info(call)
#                return BitBinaryOp(dbg, binary_pseudo_funcs[func_name], val,
#                                   amount)
#            elif func_name == 'repeat':
#                if call.keywords:
#                    raise QwertySyntaxError('Keywords arguments not '
#                                            'supported to {}()'
#                                            .format(func_name),
#                                            self.get_debug_info(call))
#                if len(call.args) != 1:
#                    raise QwertySyntaxError('{}() expects one positional '
#                                            'argument: the amount of times to '
#                                            'repeat'
#                                            .format(func_name),
#                                            self.get_debug_info(call))
#                bits = self.visit(operand)
#                amount_node = call.args[0]
#                amount = self.extract_dimvar_expr(amount_node)
#                dbg = self.get_debug_info(call)
#                return BitRepeat(dbg, bits, amount)
#            else:
#                raise QwertySyntaxError('Unknown pseudo-function {}'
#                                        .format(func_name),
#                                        self.get_debug_info(call))
#
#    def visit_Tuple(self, tup: ast.Tuple):
#        """
#        Convert a Python tuple literal to a nest of Qwerty ``BitConcat`` AST
#        nodes.
#        """
#        if not tup.elts:
#            raise QwertySyntaxError('Empty tuple not supported',
#                                    self.get_debug_info(tup))
#        cur = self.visit(tup.elts[0])
#        for elt in tup.elts[1:]:
#            dbg = self.get_debug_info(tup)
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
#                                        self.get_debug_info(sub))
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
#        dbg = self.get_debug_info(sub)
#        return Slice(dbg, val, lower, upper)
#
#    def visit(self, node: ast.AST):
#        if isinstance(node, ast.Name):
#            return self.visit_Name(node)
#        elif isinstance(node, ast.UnaryOp):
#            return self.visit_UnaryOp(node)
#        elif isinstance(node, ast.BinOp):
#            return self.visit_BinOp(node)
#        elif isinstance(node, ast.Call):
#            return self.visit_Call(node)
#        elif isinstance(node, ast.Tuple):
#            return self.visit_Tuple(node)
#        elif isinstance(node, ast.Subscript):
#            return self.visit_Subscript(node)
#        else:
#            return self.base_visit(node)

#def convert_classical_ast(module: ast.Module, filename: str = '', line_offset: int = 0,
#                          col_offset: int = 0) -> Kernel:
#    """
#    Run the ``ClassicalVisitor`` on the provided Python AST to convert to a
#    Qwerty `@classical` AST and return the result. The return value is the same
#    as ``convert_ast()`` above.
#    """
#    if not isinstance(module, ast.Module):
#        raise QwertySyntaxError('Expected top-level Module node in Python AST',
#                                None) # This should not happen
#
#    #visitor = ClassicalVisitor(filename, line_offset, col_offset)
#    #return visitor.visit_Module(module), visitor.tvs_has_explicit_value
#    raise NotImplementedError('coming soon...')
