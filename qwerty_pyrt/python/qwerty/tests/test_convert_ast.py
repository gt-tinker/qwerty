import ast
import textwrap
import unittest

from qwerty.err import QwertySyntaxError
from qwerty.convert_ast import convert_qpu_repl_input
from qwerty._qwerty_pyrt import Expr, QLit, DebugLoc

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

    def test_mystery_qubit_literal(self):
        with self.assertRaisesRegex(QwertySyntaxError, "Unknown qubit symbol"):
            actual_qw_ast = self.convert_expr("""
                'a'
            """)

    def test_repeated_qubit_literal(self):
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
