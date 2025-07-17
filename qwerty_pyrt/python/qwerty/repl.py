"""
Python glue code to implement the Qwerty REPL. You can start the Qwerty REPL
with the following command::

    python -m qwerty

This module is responsible for taking user input, parsing it into a Python AST,
converting that to a Qwerty AST, and then handing off control to
qwerty_ast::repl to actually run the code.
"""

import ast
from .convert_ast import convert_qpu_repl_input
from .err import QwertyProgrammerError
from ._qwerty_pyrt import ReplState, TypeEnv, Expr

def repl():
    """Run the Qwerty REPL."""

    state = ReplState()
    env = TypeEnv()

    while True:
        try:
            cmd = input('(qwerty) ')
        except (EOFError, KeyboardInterrupt):
            # Print a newline so the user's shell prompt is on a fresh line
            print()
            return

        try:
            qwerty = convert_qpu_repl_input(ast.parse(cmd, mode='single'))
            qwerty.typecheck_expr(env)
            # TODO: type check qwerty AST right here

        except QwertyProgrammerError as err:
            print(f'{err.kind()}: {err}')
            continue

        result = state.run(qwerty)
        print(result)
