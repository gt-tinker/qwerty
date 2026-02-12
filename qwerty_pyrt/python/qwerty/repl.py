"""
Python glue code to implement the Qwerty REPL. You can start the Qwerty REPL
with the following command::

    python -m qwerty

This module is responsible for taking user input, parsing it into a Python AST,
converting that to a Qwerty AST, and then handing off control to
qwerty_ast::repl to actually run the code.
"""

import ast
from typing import Optional
from collections.abc import Callable
from dataclasses import dataclass
from .prelude import PreludeHandle
from .err import QwertyProgrammerError
from .convert_ast import convert_qpu_repl_input, Capturer, CapturedSymbol
from .default_qpu_prelude import default_qpu_prelude
from ._qwerty_pyrt import ReplState, TypeEnv, MacroEnv, PlainQpuStmt, \
                          PlainQpuExpr, PlainClassicalFunctionDef, \
                          SparseReplState

class ReplSymbolCapturer(Capturer):
    def __init__(self):
        self.symbol_names = set()

    def register_symbol(self, symbol_name):
        self.symbol_names.add(symbol_name)

    def shadows_python_variable(self, var_name: str) -> bool:
        return False

    def capture(self, var_name: str) -> Optional[str]:
        if var_name in self.symbol_names:
            return CapturedSymbol(var_name)

        return None

@dataclass
class ReplContext:
    state: ReplState
    type_env: TypeEnv
    macro_env: MacroEnv
    capturer: ReplSymbolCapturer

    @classmethod
    def new(cls):
        return cls(ReplState(), TypeEnv(), MacroEnv(), ReplSymbolCapturer())

    def run_stmt(self, stmt_ast) -> PlainQpuExpr:
        lowered = stmt_ast.lower(self.macro_env, self.type_env)
        if isinstance(plain_ast := lowered, PlainQpuStmt):
            # This is a plain @qpu statement AST (no metaQwerty anymore)
            plain_ast.type_check_no_ret(self.type_env)
            return self.state.run(plain_ast)
        elif isinstance(classical_func_def := lowered,
                        PlainClassicalFunctionDef):
            # This is a classical lambda
            classical_func_def.type_check(self.type_env)
            self.type_env.insert_classical_func(classical_func_def)
            self.state.insert_classical_func(classical_func_def)
            self.capturer.register_symbol(classical_func_def.get_name())
            return PlainQpuExpr.trivial(None)
        else:
            ty = type(lowered)
            assert False, f"Lowering a @qpu statement produced {ty}, huh?"

    def free_value(self, val_ast: PlainQpuExpr):
        self.state.free_value(val_ast)

    def get_sparse_state(self) -> SparseReplState:
        return self.state.get_sparse_state()

    def run_prelude(self, prelude: PreludeHandle):
        for stmt in prelude._get_stmts():
            self.run_stmt(stmt)

    def run_cmd_str(self, cmd: str) -> Optional[str]:
        stripped_cmd = cmd.strip()
        if not stripped_cmd:
            # Ignore blank lines
            return None

        if stripped_cmd.startswith('%'):
            magic_cmd = stripped_cmd[1:]
            if magic_cmd == 'state':
                return self.get_sparse_state()
            else:
                raise QwertyProgrammerError(f'Unknown magic command %{magic_cmd}')
        else:
            py_ast = ast.parse(stripped_cmd, mode='single')
            stmt_ast = convert_qpu_repl_input(py_ast,
                                              capturer=self.capturer)
            result_expr_ast = self.run_stmt(stmt_ast)

            sparse_state = self.get_sparse_state()
            rendered = result_expr_ast.render(sparse_state)
            if rendered:
                ret = rendered
            else:
                ret = None
            self.free_value(result_expr_ast)
            return ret

    def run_cmd_str_catch(self, cmd: str) -> Optional[str]:
        try:
            return self.run_cmd_str(cmd)
        except SyntaxError as err:
            return f'Syntax Error: {err}'
        except QwertyProgrammerError as err:
            return f'{err.kind()}: {err}'

def repl(prompt_func: Callable[[], str] = input,
         print_func: Callable[[str], None] = print) -> int:
    """
    Run the Qwerty REPL.

    ``prompt_func`` is an analog of ``input()`` and ``print_func`` is an analog
    of ``print()``. Both arguments exist to enable unit testing.
    """
    ctx = ReplContext.new()

    try:
        ctx.run_prelude(default_qpu_prelude)
    except QwertyProgrammerError as err:
        print_func(f'Error in prelude: {err}. This is a bug.')
        # Bail out. This should never happen
        return 1

    while True:
        try:
            cmd = prompt_func('qwerty>> ')
        except (EOFError, KeyboardInterrupt):
            # Print a newline so the user's shell prompt is on a fresh line
            print_func()
            return 0

        result = ctx.run_cmd_str_catch(cmd)
        if result is not None:
            print_func(result)

def get_jupyter_kernel():
    from ipykernel.kernelbase import Kernel

    class QwertyKernel(Kernel):
        implementation = 'Qwerty'
        implementation_version = '0.0.1'
        banner = '>> >> >> QWERTY >> >> >>'
        language_info = {
            'name': 'Qwerty',
            'mimetype': 'text/x-python',
            'file_extension': '.py',
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
        }

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ctx = ReplContext.new()

            self.ctx.run_prelude(default_qpu_prelude)

        def do_execute(self, code, silent, **kwargs):
            result = self.ctx.run_cmd_str_catch(code)

            if result is not None and not silent:
                stream_content = {'name': 'stdout', 'text': result}
                self.send_response(self.iopub_socket, 'stream', stream_content)

            return {'status': 'ok',
                    'execution_count': self.execution_count,
                    'payload': [],
                    'user_expressions': {}}

    return QwertyKernel
