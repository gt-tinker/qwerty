import unittest
from unittest.mock import Mock, call

from qwerty.repl import repl

class ReplTests(unittest.TestCase):
    PROMPT = '(qwerty) '

    def test_input_initial_eof(self):
        prompt_func = Mock(side_effect=[EOFError()])
        print_func = Mock()
        repl(prompt_func, print_func)
        prompt_func.assert_called_once_with(self.PROMPT)
        print_func.assert_called_once_with()

    def test_input_zero_qubit(self):
        prompt_func = Mock(side_effect=["__SYM_STD0__()", EOFError()])
        print_func = Mock()
        repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(self.PROMPT), call(self.PROMPT)])
        print_func.assert_has_calls([call("q[0]"), call()])

    def test_input_zero_qubit_sigint(self):
        prompt_func = Mock(side_effect=["__SYM_STD0__()", KeyboardInterrupt()])
        print_func = Mock()
        repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(self.PROMPT), call(self.PROMPT)])
        print_func.assert_has_calls([call("q[0]"), call()])

    def test_input_zero_qubit_whitespace(self):
        prompt_func = Mock(side_effect=["    __SYM_STD0__()     ", EOFError()])
        print_func = Mock()
        repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(self.PROMPT), call(self.PROMPT)])
        print_func.assert_has_calls([call("q[0]"), call()])

    def test_input_empty_then_zero_qubit(self):
        prompt_func = Mock(side_effect=['    ', "__SYM_STD0__()", EOFError()])
        print_func = Mock()
        repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(self.PROMPT), call(self.PROMPT)])
        print_func.assert_has_calls([call("q[0]"), call()])

    def test_input_python_syntax_error_recovery(self):
        prompt_func = Mock(side_effect=["'0", "__SYM_STD0__()", EOFError()])
        print_func = Mock()
        repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(self.PROMPT),
                                      call(self.PROMPT),
                                      call(self.PROMPT)])
        print_func.assert_has_calls([call('Syntax Error: unterminated string '
                                          'literal (detected at line 1) '
                                          '(<unknown>, line 1)'),
                                     call("q[0]"),
                                     call()])

    def test_input_return(self):
        prompt_func = Mock(side_effect=["return __SYM_STD0__()", EOFError()])
        print_func = Mock()
        repl(prompt_func, print_func)
        prompt_func.assert_has_calls([call(self.PROMPT), call(self.PROMPT)])
        print_func.assert_has_calls([call("Type Error: The return statement "
                                          "can only be written inside a "
                                          "function. (at column 1)"), call()])
