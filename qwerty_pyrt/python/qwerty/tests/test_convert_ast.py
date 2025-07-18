import ast
import textwrap
import unittest

from qwerty.err import QwertySyntaxError
from qwerty.convert_ast import convert_qpu_repl_input
from qwerty._qwerty_pyrt import Expr, QLit, DebugLoc, Basis, Vector

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
        expected_qw_ast = Expr.new_qlit(QLit.new_zero_qubit(dbg), dbg)

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_qubit_literal_one(self):
        actual_qw_ast = self.convert_expr("""
            '1'
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = Expr.new_qlit(QLit.new_one_qubit(dbg), dbg)

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_qubit_literal_unit(self):
        actual_qw_ast = self.convert_expr("""
            ''
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = Expr.new_qlit(QLit.new_qubit_unit(dbg), dbg)

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_unknown_constant(self):
        with self.assertRaisesRegex(QwertySyntaxError, "Unknown constant"):
            actual_qw_ast = self.convert_expr("""
                None
            """)

    def test_qubit_literal_mystery_symbol(self):
        with self.assertRaisesRegex(QwertySyntaxError, "Unknown qubit symbol"):
            actual_qw_ast = self.convert_expr("""
                'a'
            """)

    def test_qubit_literal_multibit(self):
        actual_qw_ast = self.convert_expr("""
            '010'
        """)
        dbg = self.dbg(1, 1)
        expected_qw_ast = Expr.new_qlit(QLit.new_qubit_tensor([
            QLit.new_zero_qubit(dbg),
            QLit.new_one_qubit(dbg),
            QLit.new_zero_qubit(dbg),
        ], dbg), dbg)

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_basis_singleton(self):
        actual_qw_ast = self.convert_expr("""
            '010' >> -'010'
        """)
        dbg1 = self.dbg(1, 1)
        dbg2 = self.dbg(1, 11)
        dbg3 = self.dbg(1, 10)
        expected_qw_ast = Expr.new_basis_translation(
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
            dbg1)
        self.assertEqual(actual_qw_ast, expected_qw_ast)
