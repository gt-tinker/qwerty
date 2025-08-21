import ast
import textwrap
import unittest
from typing import Optional

from qwerty.runtime import bit
from qwerty.err import QwertySyntaxError
from qwerty.convert_ast import convert_qpu_repl_input, \
                               convert_classical_repl_input, Capturer, \
                               CaptureError, CapturedSymbol, CapturedValue, \
                               CapturedBitReg
from qwerty._qwerty_pyrt import QpuExpr, ClassicalExpr, UnaryOpKind, \
                                BinaryOpKind, DebugLoc, Basis, Vector, \
                                QpuStmt, ClassicalStmt, EmbedKind, FloatExpr

class SingleVarCapturer(Capturer):
    def __init__(self, name: str, captured: CapturedValue):
        self.name = name
        self.captured = captured

    def shadows_python_variable(self, var_name: str) -> bool:
        return var_name == self.name

    def capture(self, var_name: str) -> Optional[str]:
        if var_name == self.name:
            return self.captured
        else:
            return None

class BadTypeCapturer(Capturer):
    def __init__(self, name: str):
        self.name = name

    def shadows_python_variable(self, var_name: str) -> bool:
        return var_name == self.name

    def capture(self, var_name: str) -> Optional[str]:
        if var_name == self.name:
            raise CaptureError('int')
        else:
            return None

class ConvertAstQpuTests(unittest.TestCase):
    def convert_expr(self, code, *, capturer=None):
        code = textwrap.dedent(code.strip())
        py_ast = ast.parse(code, mode='single')
        return convert_qpu_repl_input(py_ast, capturer=capturer)

    def dbg(self, line, col):
        return DebugLoc('<input>', line, col)

    def test_qubit_literal_zero(self):
        actual_qw_ast = self.convert_expr("""
            '0'
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_qlit(Vector.new_vector_symbol('0', dbg)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_qubit_literal_one(self):
        actual_qw_ast = self.convert_expr("""
            '1'
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_qlit(Vector.new_vector_symbol('1', dbg)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_qubit_literal_unit(self):
        actual_qw_ast = self.convert_expr("""
            ''
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_qlit(Vector.new_vector_unit(dbg)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_unknown_constant(self):
        with self.assertRaisesRegex(QwertySyntaxError, "Unknown constant"):
            self.convert_expr("""
                None
            """)

    def test_qubit_literal_multibit(self):
        actual_qw_ast = self.convert_expr("""
            '010'
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = QpuStmt.new_expr(QpuExpr.new_qlit(
            Vector.new_vector_bi_tensor(
                Vector.new_vector_bi_tensor(
                    Vector.new_vector_symbol('0', dbg),
                    Vector.new_vector_symbol('1', dbg),
                    dbg,
                ),
                Vector.new_vector_symbol('0', dbg), dbg)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_qubit_literal_non_uniform_neg_superpos(self):
        actual_qw_ast = self.convert_expr("""
            0.25*'0' - 0.75*'1'
        """)
        dbg_neg = self.dbg(1, 1)
        dbg_strlit1 = self.dbg(1, 6)
        dbg_strlit2 = self.dbg(1, 17)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_non_uniform_superpos([
                (0.25, Vector.new_vector_symbol('0', dbg_strlit1)),
                (0.75, Vector.new_vector_tilt(
                    Vector.new_vector_symbol('1', dbg_strlit2),
                    FloatExpr.new_const(180.0, dbg_neg),
                    dbg_neg)),
            ], dbg_neg))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_basis_singleton(self):
        actual_qw_ast = self.convert_expr("""
            '010' >> -'010'
        """)
        dbg1 = self.dbg(1, 1)
        dbg2 = self.dbg(1, 11)
        dbg3 = self.dbg(1, 10)
        expected_qw_ast = QpuStmt.new_expr(QpuExpr.new_basis_translation(
            Basis.new_basis_literal([
                Vector.new_vector_bi_tensor(
                    Vector.new_vector_bi_tensor(
                        Vector.new_vector_symbol('0', dbg1),
                        Vector.new_vector_symbol('1', dbg1),
                        dbg1),
                    Vector.new_vector_symbol('0', dbg1),
                    dbg1)
                ], dbg1),
            Basis.new_basis_literal([
                Vector.new_vector_tilt(
                    Vector.new_vector_bi_tensor(
                        Vector.new_vector_bi_tensor(
                            Vector.new_vector_symbol('0', dbg2),
                            Vector.new_vector_symbol('1', dbg2),
                            dbg2),
                        Vector.new_vector_symbol('0', dbg2),
                        dbg2),
                    FloatExpr.new_const(180.0, dbg3),
                    dbg3)
            ], dbg3),
            dbg1))
        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_assign_multi_target(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "Assigning to multiple targets .*not "
                                    "supported"):
            self.convert_expr("""
                a = b = '0'
            """)

    def test_assign_empty_tgt(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "Unpacking assignment must have at least "
                                    "two names"):
            self.convert_expr("""
                () = ''
            """)

    def test_assign_single_unpack(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "Unpacking assignment must have at least "
                                    "two names"):
            self.convert_expr("""
                x, = '0'
            """)

    def test_assign_simple(self):
        actual_qw_ast = self.convert_expr("""
            x = '0'
        """)
        dbg_assign = self.dbg(1, 1)
        dbg_str = self.dbg(1, 5)
        expected_qw_ast = QpuStmt.new_assign('x', QpuExpr.new_qlit(
            Vector.new_vector_symbol('0', dbg_str)), dbg_assign)

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_assign_unpack_three(self):
        actual_qw_ast = self.convert_expr("""
            x, y, z = q
        """)
        dbg_assign = self.dbg(1, 1)
        dbg_q = self.dbg(1, 11)
        expected_qw_ast = QpuStmt.new_unpack_assign(
            ['x', 'y', 'z'],
            QpuExpr.new_variable('q', dbg_q),
            dbg_assign)

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_basis_translation_elementwise_sugar(self):
        actual_qw_ast = self.convert_expr("""
            {'0'>>'0', '1'>>-'1'}
        """)
        dbg_set = self.dbg(1, 1)
        dbg_str1 = self.dbg(1, 2)
        dbg_str2 = self.dbg(1, 7)
        dbg_str3 = self.dbg(1, 12)
        dbg_neg = self.dbg(1, 17)
        dbg_str4 = self.dbg(1, 18)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_basis_translation(
                Basis.new_basis_literal([Vector.new_vector_symbol('0', dbg_str1),
                                         Vector.new_vector_symbol('1', dbg_str3)],
                                        dbg_set),
                Basis.new_basis_literal([Vector.new_vector_symbol('0', dbg_str2),
                                         Vector.new_vector_tilt(
                                             Vector.new_vector_symbol('1', dbg_str4),
                                             FloatExpr.new_const(180.0, dbg_neg),
                                             dbg_neg)],
                                        dbg_set),
                dbg_set))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_bit_literal_4bit(self):
        actual_qw_ast = self.convert_expr("""
            bit[4](0b1101)
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_bit_literal(0b1101, 4, dbg))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_bit_literal_nonint_dim(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "must be an integer constant"):
            self.convert_expr("""
                bit[N](0b1101)
            """)

    def test_bit_literal_float_dim(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "must be an integer constant"):
            self.convert_expr("""
                bit[2.0](0b1101)
            """)

    def test_bit_literal_no_args(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "requires only constant bits .* between .* paren"):
            self.convert_expr("""
                bit[4]()
            """)

    def test_bit_literal_two_args(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "requires only constant bits .* between .* paren"):
            self.convert_expr("""
                bit[4](0b10, 0b11)
            """)

    def test_bit_literal_bits_float(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "requires only constant bits .* between .* paren"):
            self.convert_expr("""
                bit[4](3.0)
            """)

    def test_bit_literal_bits_name(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "requires only constant bits .* between .* paren"):
            self.convert_expr("""
                bit[4](x)
            """)

    def test_pipe_call_sugar_two_args(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "got 2 arguments"):
            self.convert_expr("""
                f(x, y)
            """)

    def test_pipe_call_sugar_one_arg(self):
        actual_qw_ast = self.convert_expr("""
            f(x)
        """)
        dbg_f = self.dbg(1, 1)
        dbg_x = self.dbg(1, 3)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_pipe(QpuExpr.new_variable('x', dbg_x),
                          QpuExpr.new_variable('f', dbg_f),
                          dbg_f))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_pipe_call_sugar_zero_args(self):
        actual_qw_ast = self.convert_expr("""
            f()
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_pipe(QpuExpr.new_unit_literal(dbg),
                          QpuExpr.new_variable('f', dbg),
                          dbg))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_intrinsic_measure(self):
        actual_qw_ast = self.convert_expr("""
            __MEASURE__({'0','1'})
        """)
        dbg_call = self.dbg(1, 1)
        dbg_set = self.dbg(1, 13)
        dbg_str1 = self.dbg(1, 14)
        dbg_str2 = self.dbg(1, 18)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_measure(
                Basis.new_basis_literal([Vector.new_vector_symbol('0', dbg_str1),
                                         Vector.new_vector_symbol('1', dbg_str2)],
                                        dbg_set),
                dbg_call))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_intrinsic_measure_no_args(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "Wrong number of arguments to intrinsic"):
            self.convert_expr("""
                __MEASURE__()
            """)

    def test_intrinsic_measure_two_args(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "Wrong number of arguments to intrinsic"):
            self.convert_expr("""
                __MEASURE__({}, {})
            """)

    def test_intrinsic_discard(self):
        actual_qw_ast = self.convert_expr("""
            __DISCARD__()
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_discard(dbg))
        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_intrinsic_embed_xor(self):
        capturer = SingleVarCapturer("bubba", CapturedSymbol('bubba__mangled'))
        actual_qw_ast = self.convert_expr("""
            __EMBED_XOR__(bubba)
        """, capturer=capturer)
        dbg_call = self.dbg(1, 1)
        dbg_arg = self.dbg(1, 15)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_embed_classical(
                QpuExpr.new_variable("bubba__mangled", dbg_arg),
                EmbedKind.Xor,
                dbg_call))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_intrinsic_embed_sign(self):
        capturer = SingleVarCapturer("bubba", CapturedSymbol('bubba__mangled'))
        actual_qw_ast = self.convert_expr("""
            __EMBED_SIGN__(bubba)
        """, capturer=capturer)
        dbg_call = self.dbg(1, 1)
        dbg_arg = self.dbg(1, 16)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_embed_classical(
                QpuExpr.new_variable("bubba__mangled", dbg_arg),
                EmbedKind.Sign,
                dbg_call))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_intrinsic_embed_inplace(self):
        capturer = SingleVarCapturer("bubba", CapturedSymbol('bubba__mangled'))
        actual_qw_ast = self.convert_expr("""
            __EMBED_INPLACE__(bubba)
        """, capturer=capturer)
        dbg_call = self.dbg(1, 1)
        dbg_arg = self.dbg(1, 19)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_embed_classical(
                QpuExpr.new_variable("bubba__mangled", dbg_arg),
                EmbedKind.InPlace,
                dbg_call))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_intrinsic_embed_badly_typed_func(self):
        capturer = BadTypeCapturer("bubba")
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "cannot be used"):
            self.convert_expr("""
                __EMBED_XOR__(bubba)
            """, capturer=capturer)

    def test_unit_literal(self):
        actual_qw_ast = self.convert_expr("""
            []
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_unit_literal(dbg))
        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_unit_literal_nonempty(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "list literals are not supported"):
            self.convert_expr("""
                [1,2,3]
            """)

    def test_captured_bit_reg(self):
        capturer = SingleVarCapturer("bubba", CapturedBitReg(bit[4](0b1101)))
        actual_qw_ast = self.convert_expr("""
            bubba
        """, capturer=capturer)
        dbg = self.dbg(1, 1)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_bit_literal(0b1101, 4, dbg))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_qubit_literal_zero_intrinsic(self):
        actual_qw_ast = self.convert_expr("""
            __SYM_STD0__()
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = QpuStmt.new_expr(
            QpuExpr.new_qlit(Vector.new_zero_vector(dbg)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_basis_singleton_zero_intrinsic(self):
        actual_qw_ast = self.convert_expr("""
            __SYM_STD0__() >> __SYM_STD0__()
        """)
        dbg_call1 = self.dbg(1, 1)
        dbg_call2 = self.dbg(1, 19)
        expected_qw_ast = QpuStmt.new_expr(QpuExpr.new_basis_translation(
            Basis.new_basis_literal([
                Vector.new_zero_vector(dbg_call1)],
                dbg_call1),
            Basis.new_basis_literal([
                Vector.new_zero_vector(dbg_call2)],
                dbg_call2),
            dbg_call1))
        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_basis_translation_zero_intrinsic(self):
        actual_qw_ast = self.convert_expr("""
            {__SYM_STD0__()} >> {__SYM_STD0__()}
        """)
        dbg_set1 = self.dbg(1, 1)
        dbg_call1 = self.dbg(1, 2)
        dbg_set2 = self.dbg(1, 21)
        dbg_call2 = self.dbg(1, 22)
        expected_qw_ast = QpuStmt.new_expr(QpuExpr.new_basis_translation(
            Basis.new_basis_literal([
                Vector.new_zero_vector(dbg_call1)],
                dbg_set1),
            Basis.new_basis_literal([
                Vector.new_zero_vector(dbg_call2)],
                dbg_set2),
            dbg_set1))
        self.assertEqual(actual_qw_ast, expected_qw_ast)

class ConvertAstClassicalTests(unittest.TestCase):
    def convert_expr(self, code, *, capturer=None):
        code = textwrap.dedent(code.strip())
        py_ast = ast.parse(code, mode='single')
        return convert_classical_repl_input(py_ast, capturer=capturer)

    def dbg(self, line, col):
        return DebugLoc('<input>', line, col)

    def test_qubit_literal_zero(self):
        actual_qw_ast = self.convert_expr("""
            ~x
        """)
        dbg_neg = self.dbg(1, 1)
        dbg_var = self.dbg(1, 2)
        expected_qw_ast = ClassicalStmt.new_expr(
            ClassicalExpr.new_unary_op(
                UnaryOpKind.Not,
                ClassicalExpr.new_variable("x", dbg_var),
                dbg_neg))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_bit_literal_4bit(self):
        actual_qw_ast = self.convert_expr("""
            bit[4](0b1101)
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = ClassicalStmt.new_expr(
            ClassicalExpr.new_bit_literal(0b1101, 4, dbg))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_bit_literal_nonint_dim(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "must be an integer constant"):
            self.convert_expr("""
                bit[N](0b1101)
            """)

    def test_bit_literal_float_dim(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "must be an integer constant"):
            self.convert_expr("""
                bit[2.0](0b1101)
            """)

    def test_bit_literal_no_args(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "requires only constant bits .* between .* paren"):
            self.convert_expr("""
                bit[4]()
            """)

    def test_bit_literal_two_args(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "requires only constant bits .* between .* paren"):
            self.convert_expr("""
                bit[4](0b10, 0b11)
            """)

    def test_bit_literal_bits_float(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "requires only constant bits .* between .* paren"):
            self.convert_expr("""
                bit[4](3.0)
            """)

    def test_bit_literal_bits_name(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "requires only constant bits .* between .* paren"):
            self.convert_expr("""
                bit[4](x)
            """)

    def test_xor_reduce(self):
        actual_qw_ast = self.convert_expr("""
            x.xor_reduce()
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = ClassicalStmt.new_expr(
            ClassicalExpr.new_reduce_op(
                BinaryOpKind.Xor,
                ClassicalExpr.new_variable("x", dbg),
                dbg))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_and_reduce(self):
        actual_qw_ast = self.convert_expr("""
            x.and_reduce()
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = ClassicalStmt.new_expr(
            ClassicalExpr.new_reduce_op(
                BinaryOpKind.And,
                ClassicalExpr.new_variable("x", dbg),
                dbg))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_or_reduce(self):
        actual_qw_ast = self.convert_expr("""
            x.or_reduce()
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = ClassicalStmt.new_expr(
            ClassicalExpr.new_reduce_op(
                BinaryOpKind.Or,
                ClassicalExpr.new_variable("x", dbg),
                dbg))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_xor_reduce_argument(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "Arguments cannot be passed to a "
                                    "reduction"):
            self.convert_expr("""
                x.xor_reduce(0)
            """)

    def test_non_pseudo_func_call(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    r"to be of the form expression\.FUNC\(\), but"):
            self.convert_expr("""
                f(x, y)
            """)

    def test_bitwise_and(self):
        actual_qw_ast = self.convert_expr("""
            x & y
        """)
        dbg_x = self.dbg(1, 1)
        dbg_y = self.dbg(1, 5)
        expected_qw_ast = ClassicalStmt.new_expr(
            ClassicalExpr.new_binary_op(
                BinaryOpKind.And,
                ClassicalExpr.new_variable("x", dbg_x),
                ClassicalExpr.new_variable("y", dbg_y),
                dbg_x))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_bitwise_or(self):
        actual_qw_ast = self.convert_expr("""
            x | y
        """)
        dbg_x = self.dbg(1, 1)
        dbg_y = self.dbg(1, 5)
        expected_qw_ast = ClassicalStmt.new_expr(
            ClassicalExpr.new_binary_op(
                BinaryOpKind.Or,
                ClassicalExpr.new_variable("x", dbg_x),
                ClassicalExpr.new_variable("y", dbg_y),
                dbg_x))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_bitwise_xor(self):
        actual_qw_ast = self.convert_expr("""
            x ^ y
        """)
        dbg_x = self.dbg(1, 1)
        dbg_y = self.dbg(1, 5)
        expected_qw_ast = ClassicalStmt.new_expr(
            ClassicalExpr.new_binary_op(
                BinaryOpKind.Xor,
                ClassicalExpr.new_variable("x", dbg_x),
                ClassicalExpr.new_variable("y", dbg_y),
                dbg_x))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_binop_unknown(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "Unknown binary operation FloorDiv"):
            self.convert_expr("""
                x // y
            """)

    def test_captured_bit_reg(self):
        capturer = SingleVarCapturer("bubba", CapturedBitReg(bit[4](0b1101)))
        actual_qw_ast = self.convert_expr("""
            bubba
        """, capturer=capturer)
        dbg = self.dbg(1, 1)
        expected_qw_ast = ClassicalStmt.new_expr(
            ClassicalExpr.new_bit_literal(0b1101, 4, dbg))

        self.assertEqual(actual_qw_ast, expected_qw_ast)
