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
            Expr.new_qlit(QLit.new_zero_qubit(dbg), dbg), dbg)

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_qubit_literal_one(self):
        actual_qw_ast = self.convert_expr("""
            '1'
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = Stmt.new_expr(
            Expr.new_qlit(QLit.new_one_qubit(dbg), dbg), dbg)

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_qubit_literal_unit(self):
        actual_qw_ast = self.convert_expr("""
            ''
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = Stmt.new_expr(
            Expr.new_qlit(QLit.new_qubit_unit(dbg), dbg), dbg)

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
        ], dbg), dbg), dbg)

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
            dbg1), dbg1)
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
            QLit.new_zero_qubit(dbg_str), dbg_str), dbg_assign)

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
                dbg_set),
            dbg_set)

        self.assertEqual(actual_qw_ast, expected_qw_ast)
