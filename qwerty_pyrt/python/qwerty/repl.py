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
from dataclasses import dataclass
from .convert_ast import convert_qpu_repl_input
from .err import QwertyProgrammerError
from .default_qpu_prelude import default_qpu_prelude
from ._qwerty_pyrt import ReplState, TypeEnv, MacroEnv

@dataclass
class ReplContext:
    state: ReplState
    type_env: TypeEnv
    macro_env: MacroEnv

    @classmethod
    def new(cls):
        return cls(ReplState(), TypeEnv(), MacroEnv())

    def run_stmt(self, stmt_ast):
        plain_ast = stmt_ast.lower(self.macro_env, self.type_env)
        plain_ast.type_check_no_ret(self.type_env)
        # TODO: Assignment statements should also update TypeEnv (not sure of
        #       the cleanest way to do this)
        return self.state.run(plain_ast)

def repl(prompt_func: Callable[[], str] = input,
         print_func: Callable[[str], None] = print) -> None:
    """
    Run the Qwerty REPL.

    ``prompt_func`` is an analog of ``input()`` and ``print_func`` is an analog
    of ``print()``. Both arguments exist to enable unit testing.
    """
    ctx = ReplContext.new()

    for stmt in default_qpu_prelude._get_stmts():
        try:
            ctx.run_stmt(stmt)
        except QwertyProgrammerError as err:
            print_func(f'Error in prelude: {err}. This is a bug.')
            # Bail out. This should never happen
            return

    while True:
        try:
            cmd = prompt_func('qwerty>> ')
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
            result_expr_ast = ctx.run_stmt(stmt_ast)
        except QwertyProgrammerError as err:
            print_func(f'{err.kind()}: {err}')
            continue

        print_func(str(result_expr_ast))
