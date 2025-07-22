import ast
import textwrap
import unittest

from qwerty.err import QwertySyntaxError
from qwerty.convert_ast import convert_qpu_repl_input
from qwerty._qwerty_pyrt import Expr, QLit, DebugLoc, Basis, Vector, Stmt

class ConvertAstTests(unittest.TestCase):
    def convert_expr(self, code):
        code = textwrap.dedent(code.strip())
        py_ast = ast.parse(code, mode='single')
        return convert_qpu_repl_input(py_ast)

    def dbg(self, line, col):
        return DebugLoc('<input>', line, col)

    def test_qubit_literal_zero(self):
        actual_qw_ast = self.convert_expr("""
            '0'
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = Stmt.new_expr(
            Expr.new_qlit(QLit.new_zero_qubit(dbg)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_qubit_literal_one(self):
        actual_qw_ast = self.convert_expr("""
            '1'
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = Stmt.new_expr(
            Expr.new_qlit(QLit.new_one_qubit(dbg)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_qubit_literal_unit(self):
        actual_qw_ast = self.convert_expr("""
            ''
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = Stmt.new_expr(
            Expr.new_qlit(QLit.new_qubit_unit(dbg)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_unknown_constant(self):
        with self.assertRaisesRegex(QwertySyntaxError, "Unknown constant"):
            self.convert_expr("""
                None
            """)

    def test_qubit_literal_mystery_symbol(self):
        with self.assertRaisesRegex(QwertySyntaxError, "Unknown qubit symbol"):
            self.convert_expr("""
                'a'
            """)

    def test_qubit_literal_multibit(self):
        actual_qw_ast = self.convert_expr("""
            '010'
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = Stmt.new_expr(Expr.new_qlit(QLit.new_qubit_tensor([
            QLit.new_zero_qubit(dbg),
            QLit.new_one_qubit(dbg),
            QLit.new_zero_qubit(dbg),
        ], dbg)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_basis_singleton(self):
        actual_qw_ast = self.convert_expr("""
            '010' >> -'010'
        """)
        dbg1 = self.dbg(1, 1)
        dbg2 = self.dbg(1, 11)
        dbg3 = self.dbg(1, 10)
        expected_qw_ast = Stmt.new_expr(Expr.new_basis_translation(
            Basis.new_basis_literal([
                Vector.new_vector_tensor([
                    Vector.new_zero_vector(dbg1),
                    Vector.new_one_vector(dbg1),
                    Vector.new_zero_vector(dbg1),
                ], dbg1)], dbg1),
            Basis.new_basis_literal([
                Vector.new_vector_tilt(
                    Vector.new_vector_tensor([
                        Vector.new_zero_vector(dbg2),
                        Vector.new_one_vector(dbg2),
                        Vector.new_zero_vector(dbg2),
                    ], dbg2),
                    180.0,
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
        expected_qw_ast = Stmt.new_assign('x', Expr.new_qlit(
            QLit.new_zero_qubit(dbg_str)), dbg_assign)

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_assign_unpack_three(self):
        actual_qw_ast = self.convert_expr("""
            x, y, z = q
        """)
        dbg_assign = self.dbg(1, 1)
        dbg_q = self.dbg(1, 11)
        expected_qw_ast = Stmt.new_unpack_assign(
            ['x', 'y', 'z'],
            Expr.new_variable('q', dbg_q),
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
        expected_qw_ast = Stmt.new_expr(
            Expr.new_basis_translation(
                Basis.new_basis_literal([Vector.new_zero_vector(dbg_str1),
                                         Vector.new_one_vector(dbg_str3)],
                                        dbg_set),
                Basis.new_basis_literal([Vector.new_zero_vector(dbg_str2),
                                         Vector.new_vector_tilt(
                                             Vector.new_one_vector(dbg_str4),
                                             180.0,
                                             dbg_neg)],
                                        dbg_set),
                dbg_set))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_bit_literal_4bit(self):
        actual_qw_ast = self.convert_expr("""
            bit[4](0b1101)
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = Stmt.new_expr(
            Expr.new_bit_literal(4, 0b1101, dbg))

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
        expected_qw_ast = Stmt.new_expr(
            Expr.new_pipe(Expr.new_variable('x', dbg_x),
                          Expr.new_variable('f', dbg_f),
                          dbg_f))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_pipe_call_sugar_zero_args(self):
        actual_qw_ast = self.convert_expr("""
            f()
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = Stmt.new_expr(
            Expr.new_pipe(Expr.new_unit_literal(dbg),
                          Expr.new_variable('f', dbg),
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
        expected_qw_ast = Stmt.new_expr(
            Expr.new_measure(
                Basis.new_basis_literal([Vector.new_zero_vector(dbg_str1),
                                         Vector.new_one_vector(dbg_str2)],
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
        expected_qw_ast = Stmt.new_expr(Expr.new_discard(dbg))
        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_unit_literal(self):
        actual_qw_ast = self.convert_expr("""
            []
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = Stmt.new_expr(Expr.new_unit_literal(dbg))
        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_unit_literal_nonempty(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "list literals are not supported"):
            self.convert_expr("""
                [1,2,3]
            """)
