"""
Python glue code to implement the Qwerty REPL. You can start the Qwerty REPL
with the following command::

    python -m qwerty

This module is responsible for taking user input, parsing it into a Python AST,
converting that to a Qwerty AST, and then handing off control to
qwerty_ast::repl to actually run the code.
"""

import ast
from collections.abc import Callable
from .convert_ast import convert_qpu_repl_input
from .err import QwertyProgrammerError
from ._qwerty_pyrt import ReplState, TypeEnv

def repl(prompt_func: Callable[[], str] = input,
         print_func: Callable[[str], None] = print) -> None:
    """
    Run the Qwerty REPL.

    ``prompt_func`` is an analog of ``input()`` and ``print_func`` is an analog
    of ``print()``. Both arguments exist to enable unit testing.
    """
    state = ReplState()
    env = TypeEnv()

    while True:
        try:
            cmd = prompt_func('(qwerty) ')
        except (EOFError, KeyboardInterrupt):
            # Print a newline so the user's shell prompt is on a fresh line
            print_func()
            return


        stripped_cmd = cmd.strip()
        if not stripped_cmd:
            # Ignore blank lines
            continue

        try:
            py_ast = ast.parse(stripped_cmd, mode='single')
        except SyntaxError as err:
            print_func(f'Syntax Error: {err}')
            continue

        try:
            stmt_ast = convert_qpu_repl_input(py_ast)
            plain_ast = stmt_ast.extract()
            plain_ast.type_check_no_ret(env)
        except QwertyProgrammerError as err:
            print_func(f'{err.kind()}: {err}')
            continue

        result_expr_ast = state.run(plain_ast)
        print_func(str(result_expr_ast))
