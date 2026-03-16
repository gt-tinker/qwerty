import unittest
from unittest.mock import Mock, call

from qwerty.repl import repl

PROMPT = 'qwerty>> '

class ReplTests(unittest.TestCase):
    def test_input_initial_eof(self):
        prompt_func = Mock(side_effect=[EOFError()])
        print_func = Mock()
        exit_code = repl(prompt_func, print_func)
        prompt_func.assert_called_once_with(PROMPT)
        print_func.assert_called_once_with()
        self.assertEqual(exit_code, 0)

    def test_input_zero_qubit(self):
        prompt_func = Mock(side_effect=["'0'", EOFError()])
        print_func = Mock()
        exit_code = repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(PROMPT), call(PROMPT)])
        print_func.assert_has_calls([call("'0'"), call()])
        self.assertEqual(exit_code, 0)

    def test_input_zero_qubit_sigint(self):
        prompt_func = Mock(side_effect=["'0'", KeyboardInterrupt()])
        print_func = Mock()
        exit_code = repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(PROMPT), call(PROMPT)])
        print_func.assert_has_calls([call("'0'"), call()])
        self.assertEqual(exit_code, 0)

    def test_input_zero_qubit_whitespace(self):
        prompt_func = Mock(side_effect=["    '0'     ", EOFError()])
        print_func = Mock()
        exit_code = repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(PROMPT), call(PROMPT)])
        print_func.assert_has_calls([call("'0'"), call()])
        self.assertEqual(exit_code, 0)

    def test_input_empty_then_zero_qubit(self):
        prompt_func = Mock(side_effect=['    ', "'0'", EOFError()])
        print_func = Mock()
        exit_code = repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(PROMPT), call(PROMPT)])
        print_func.assert_has_calls([call("'0'"), call()])
        self.assertEqual(exit_code, 0)

    def test_input_python_syntax_error_recovery(self):
        prompt_func = Mock(side_effect=["'0", "'0'", EOFError()])
        print_func = Mock()
        exit_code = repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(PROMPT),
                                      call(PROMPT),
                                      call(PROMPT)])
        print_func.assert_has_calls([call('Syntax Error: unterminated string '
                                          'literal (detected at line 1) '
                                          '(<unknown>, line 1)'),
                                     call("'0'"),
                                     call()])
        self.assertEqual(exit_code, 0)

    def test_input_return(self):
        prompt_func = Mock(side_effect=["return '0'", EOFError()])
        print_func = Mock()
        exit_code = repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(PROMPT), call(PROMPT)])
        print_func.assert_has_calls([call("Type Error: The return statement "
                                          "can only be written inside a "
                                          "function. (at column 1)"), call()])
        self.assertEqual(exit_code, 0)

    def test_cfunc_def(self):
        prompt_func = Mock(side_effect=["f: cfunc = lambda x: ~x", "f.sign('0')", EOFError()])
        print_func = Mock()
        exit_code = repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(PROMPT), call(PROMPT), call(PROMPT)])
        print_func.assert_has_calls([call("-'0'"), call()])
        self.assertEqual(exit_code, 0)

class AnyOf:
    """
    Based on ``unittest.mock._ANY`` in the CPython source code. Matches any
    of the arguments passed.
    """
    def __init__(self, *calls):
        self.calls = calls

    def __eq__(self, other):
        return any(c == other for c in self.calls)

    def __ne__(self, other):
        return not (self == other)

class CrnchSummit2026PosterReplTests(unittest.TestCase):
    def test_poster(self):
        inputs_and_outputs = [
            ("measure('0')", call('bit(0)')),
            ("measure('0')", call('bit(0)')),
            ("measure('0')", call('bit(0)')),
            ("measure('0')", call('bit(0)')),
            ("measure('1')", call('bit(1)')),
            ("measure('1')", call('bit(1)')),
            ("measure('1')", call('bit(1)')),
            ("measure('1')", call('bit(1)')),
            ("measure('0'+'1')", AnyOf(call('bit(0)'), call('bit(1)'))),
            ("measure('0'+'1')", AnyOf(call('bit(0)'), call('bit(1)'))),
            ("measure('0'+'1')", AnyOf(call('bit(0)'), call('bit(1)'))),
            ("measure('0'+'1')", AnyOf(call('bit(0)'), call('bit(1)'))),
            ("measure(-'0')", call('bit(0)')),
            ("measure(-'0')", call('bit(0)')),
            ("measure(-'0')", call('bit(0)')),
            ("measure(-'0')", call('bit(0)')),
            ("measure(-'1')", call('bit(1)')),
            ("measure(-'1')", call('bit(1)')),
            ("measure(-'1')", call('bit(1)')),
            ("measure(-'1')", call('bit(1)')),
            ("measure('0'-'1')", AnyOf(call('bit(0)'), call('bit(1)'))),
            ("measure('0'-'1')", AnyOf(call('bit(0)'), call('bit(1)'))),
            ("measure('0'-'1')", AnyOf(call('bit(0)'), call('bit(1)'))),
            ("measure('0'-'1')", AnyOf(call('bit(0)'), call('bit(1)'))),
            ("flip = {'0'>>'1', '1'>>'0'}", None),
            ("flip('0')", call("'1'")),
            ("flip('1')", call("'0'")),
            ("flip('0'-'1')", call("(-'0') + ('1')")),
            ("flop = {'0'+'1'>>'0', '0'-'1'>>'1'}", None),
            ("flop('0'+'1')", call("'0'")),
            ("flop('0'-'1')", call("'1'")),
            ("f: cfunc = lambda x: x", None),
            ("f.sign('0')", call("'0'")),
            ("f.sign('1')", call("-'1'")),
            ("f.sign('0'+'1')", call("('0') + (-'1')")),
            ("'0' | flip", call("'1'")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(1)")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(1)")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(1)")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(1)")),
            ("'0'+'1' | f.sign", call("('0') + (-'1')")),
            ("'0'-'1' | flop", call("'1'")),
            ("f: cfunc = lambda x: ~x", None),
            ("'0'+'1' | f.sign | flop | measure", call("bit(1)")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(1)")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(1)")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(1)")),
            ("f: cfunc = lambda x: bit(0)", None),
            ("'0'+'1' | f.sign | flop | measure", call("bit(0)")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(0)")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(0)")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(0)")),
            ("f: cfunc = lambda x: bit(1)", None),
            ("'0'+'1' | f.sign | flop | measure", call("bit(0)")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(0)")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(0)")),
            ("'0'+'1' | f.sign | flop | measure", call("bit(0)")),
            (EOFError(), call()),
        ]
        prompt_func_results, prints = zip(*inputs_and_outputs)
        prompt_func_calls = [call(PROMPT)]*len(prompt_func_results)
        print_func_calls = [p for p in prints if p is not None]

        prompt_func = Mock(side_effect=prompt_func_results)
        print_func = Mock()
        exit_code = repl(prompt_func, print_func)
        prompt_func.assert_has_calls(prompt_func_calls)
        print_func.assert_has_calls(print_func_calls)
        self.assertEqual(exit_code, 0)
