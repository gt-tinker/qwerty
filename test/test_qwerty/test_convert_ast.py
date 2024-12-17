import ast
import math
import textwrap
import unittest

from qwerty.err import QwertySyntaxError
from qwerty.convert_ast import convert_qpu_expr
from qwerty._qwerty_harness import Pipe, DebugInfo, QubitLiteral, DimVarExpr, \
                                   BuiltinBasis, Measure, Identity, Discard, \
                                   Flip, BiTensor, Variable, BroadcastTensor, \
                                   Instantiate, BasisTranslation, Pred, \
                                   Phase, Adjoint, FloatLiteral, \
                                   FloatBinaryOp, \
                                   PLUS, MINUS, X, Y, Z, FOURIER, \
                                   FLOAT_MUL, FLOAT_DIV, FLOAT_ADD

class ConvertAstTests(unittest.TestCase):
    filename, line_off, col_off = "bubba.py", 3, 4

    def convert_expr(self, code):
        code = textwrap.dedent(code.strip())
        py_ast = ast.parse(code, mode='eval')
        return convert_qpu_expr(py_ast, self.filename, self.line_off,
                                self.col_off, no_pyframe=True)

    def dbg(self, line, col):
        return DebugInfo(self.filename, self.line_off+line, self.col_off+col, None)

    def dim(self, offset):
        return DimVarExpr('', offset)

    def test_unknown_constant(self):
        with self.assertRaisesRegex(QwertySyntaxError, "Unknown constant"):
            actual_qw_ast = self.convert_expr("""
                None
            """)

    def test_empty_qubit_literal(self):
        with self.assertRaisesRegex(QwertySyntaxError, "empty"):
            actual_qw_ast = self.convert_expr("""
                ''
            """)

    def test_mystery_qubit_literal(self):
        with self.assertRaisesRegex(QwertySyntaxError, "Unknown state"):
            actual_qw_ast = self.convert_expr("""
                'a'
            """)

    def test_qubit_literal(self):
        actual_qw_ast = self.convert_expr("""
            'p'
        """)
        expected_qw_ast = \
            QubitLiteral(
                self.dbg(1, 1),
                PLUS,
                X,
                self.dim(1))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_repeated_qubit_literal(self):
        actual_qw_ast = self.convert_expr("""
            'pppp'
        """)
        expected_qw_ast = \
            QubitLiteral(
                self.dbg(1, 1),
                PLUS,
                X,
                self.dim(4))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_qubit_literal_with_diff_eigenstates(self):
        actual_qw_ast = self.convert_expr("""
            'mp'
        """)
        expected_qw_ast = \
            BiTensor(
                self.dbg(1, 1),
                QubitLiteral(
                    self.dbg(1, 1),
                    MINUS,
                    X,
                    self.dim(1)),
                QubitLiteral(
                    self.dbg(1, 1),
                    PLUS,
                    X,
                    self.dim(1)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_qubit_literal_with_multiple_primitive_bases(self):
        actual_qw_ast = self.convert_expr("""
            'p0im1j'
        """)
        expected_qw_ast = \
            BiTensor(
                self.dbg(1, 1),
                BiTensor(
                    self.dbg(1, 1),
                    BiTensor(
                        self.dbg(1, 1),
                        BiTensor(
                            self.dbg(1, 1),
                            BiTensor(
                                self.dbg(1, 1),
                                QubitLiteral(
                                    self.dbg(1, 1),
                                    PLUS,
                                    X,
                                    self.dim(1)),
                                QubitLiteral(
                                    self.dbg(1, 1),
                                    PLUS,
                                    Z,
                                    self.dim(1))),
                            QubitLiteral(
                                self.dbg(1, 1),
                                PLUS,
                                Y,
                                self.dim(1))),
                        QubitLiteral(
                            self.dbg(1, 1),
                            MINUS,
                            X,
                            self.dim(1))),
                    QubitLiteral(
                        self.dbg(1, 1),
                        MINUS,
                        Z,
                        self.dim(1))),
                QubitLiteral(
                    self.dbg(1, 1),
                    MINUS,
                    Y,
                    self.dim(1)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_id(self):
        actual_qw_ast = self.convert_expr("""
            id
        """)
        expected_qw_ast = Identity(
            self.dbg(1, 1),
            self.dim(1))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_discard(self):
        actual_qw_ast = self.convert_expr("""
            discard
        """)
        expected_qw_ast = Discard(
            self.dbg(1, 1),
            self.dim(1))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_measure(self):
        actual_qw_ast = self.convert_expr("""
            measure
        """)
        expected_qw_ast = Measure(
            self.dbg(1, 1),
            BuiltinBasis(
                self.dbg(1, 1),
                Z,
                self.dim(1)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_flip(self):
        actual_qw_ast = self.convert_expr("""
            flip
        """)
        expected_qw_ast = Flip(
            self.dbg(1, 1),
            BuiltinBasis(
                self.dbg(1, 1),
                Z,
                self.dim(1)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_std(self):
        actual_qw_ast = self.convert_expr("""
            std
        """)
        expected_qw_ast = BuiltinBasis(
            self.dbg(1, 1),
            Z,
            self.dim(1))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_pm(self):
        actual_qw_ast = self.convert_expr("""
            pm
        """)
        expected_qw_ast = BuiltinBasis(
            self.dbg(1, 1),
            X,
            self.dim(1))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_ij(self):
        actual_qw_ast = self.convert_expr("""
            ij
        """)
        expected_qw_ast = BuiltinBasis(
            self.dbg(1, 1),
            Y,
            self.dim(1))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_bare_fourier(self):
        with self.assertRaisesRegex(QwertySyntaxError, r"fourier\[1\]"):
            actual_qw_ast = self.convert_expr("""
                fourier
            """)

    def test_varname(self):
        actual_qw_ast = self.convert_expr("""
            x
        """)
        expected_qw_ast = Variable(
            self.dbg(1, 1),
            "x")

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_fourier3(self):
        actual_qw_ast = self.convert_expr("""
            fourier[3]
        """)
        expected_qw_ast = BuiltinBasis(
            self.dbg(1, 1),
            FOURIER,
            self.dim(3))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_pm3(self):
        actual_qw_ast = self.convert_expr("""
            pm[3]
        """)
        expected_qw_ast = BroadcastTensor(
            self.dbg(1, 1),
            BuiltinBasis(
                self.dbg(1, 1),
                X,
                self.dim(1)),
            self.dim(3))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_instant(self):
        actual_qw_ast = self.convert_expr("""
            f[[3]]
        """)
        expected_qw_ast = Instantiate(
            self.dbg(1, 1),
            Variable(
                self.dbg(1, 1),
                "f"),
            [self.dim(3)])

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_bitensor(self):
        actual_qw_ast = self.convert_expr("""
            x + y
        """)
        expected_qw_ast = BiTensor(
            self.dbg(1, 1),
            Variable(
                self.dbg(1, 1),
                "x"),
            Variable(
                self.dbg(1, 5),
                "y"))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_pipe(self):
        actual_qw_ast = self.convert_expr("""
            x | y
        """)
        expected_qw_ast = Pipe(
            self.dbg(1, 1),
            Variable(
                self.dbg(1, 1),
                "x"),
            Variable(
                self.dbg(1, 5),
                "y"))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_btrans(self):
        actual_qw_ast = self.convert_expr("""
            std >> pm
        """)
        expected_qw_ast = BasisTranslation(
            self.dbg(1, 1),
            BuiltinBasis(
                self.dbg(1, 1),
                Z,
                self.dim(1)),
            BuiltinBasis(
                self.dbg(1, 8),
                X,
                self.dim(1)))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_pred(self):
        actual_qw_ast = self.convert_expr("""
            '1' & f
        """)
        expected_qw_ast = Pred(
            self.dbg(1, 1),
            QubitLiteral(
                self.dbg(1, 1),
                MINUS,
                Z,
                self.dim(1)),
            Variable(
                self.dbg(1, 7),
                'f'))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_phase_rad(self):
        actual_qw_ast = self.convert_expr("""
            f @ rad(theta)
        """)
        expected_qw_ast = Phase(
            self.dbg(1, 1),
            Variable(
                self.dbg(1, 9),
                'theta'),
            Variable(
                self.dbg(1, 1),
                'f'))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_phase_deg(self):
        actual_qw_ast = self.convert_expr("""
            f @ deg(theta)
        """)
        expected_qw_ast = Phase(
            self.dbg(1, 1),
            FloatBinaryOp(
                self.dbg(1, 5),
                FLOAT_MUL,
                FloatBinaryOp(
                    self.dbg(1, 5),
                    FLOAT_DIV,
                    Variable(
                        self.dbg(1, 9),
                        'theta'),
                    FloatLiteral(
                        self.dbg(1, 5),
                        360.0)),
                FloatLiteral(
                    self.dbg(1, 5),
                    2.0*math.pi)),
            Variable(
                self.dbg(1, 1),
                'f'))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_phase(self):
        actual_qw_ast = self.convert_expr("""
            f @ (35+10)
        """)
        expected_qw_ast = Phase(
            self.dbg(1, 1),
            FloatBinaryOp(
                self.dbg(1, 6),
                FLOAT_MUL,
                FloatBinaryOp(
                    self.dbg(1, 6),
                    FLOAT_DIV,
                    FloatBinaryOp(
                        self.dbg(1, 6),
                        FLOAT_ADD,
                        FloatLiteral(
                            self.dbg(1, 6),
                            35.0),
                        FloatLiteral(
                            self.dbg(1, 9),
                            10.0)),
                    FloatLiteral(
                        self.dbg(1, 6),
                        360.0)),
                FloatLiteral(
                    self.dbg(1, 6),
                    2.0*math.pi)),
            Variable(
                self.dbg(1, 1),
                'f'))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_phase_deg_kwarg(self):
        with self.assertRaisesRegex(QwertySyntaxError, "Keyword arguments"):
            actual_qw_ast = self.convert_expr("""
                f @ deg(theta=theta)
            """)

    def test_phase_2args(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "Wrong number of arguments 2"):
            actual_qw_ast = self.convert_expr("""
                f @ deg(theta, phi)
            """)

    def test_phase_0args(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "Wrong number of arguments 0"):
            actual_qw_ast = self.convert_expr("""
                f @ deg()
            """)

    def test_mystery_binop(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "Unknown binary operation"):
            actual_qw_ast = self.convert_expr("""
                x^y
            """)

    def test_neg(self):
        actual_qw_ast = self.convert_expr("""
            -f
        """)
        expected_qw_ast = Phase(
            self.dbg(1, 1),
            FloatLiteral(
                self.dbg(1, 1),
                math.pi),
            Variable(
                self.dbg(1, 2),
                'f'))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_adjoint(self):
        actual_qw_ast = self.convert_expr("""
            ~f
        """)
        expected_qw_ast = Adjoint(
            self.dbg(1, 1),
            Variable(
                self.dbg(1, 2),
                'f'))

        self.assertEqual(actual_qw_ast, expected_qw_ast)

    def test_mystery_unop(self):
        with self.assertRaisesRegex(QwertySyntaxError,
                                    "Unknown unary operation"):
            actual_qw_ast = self.convert_expr("""
                not x
            """)
