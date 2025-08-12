from collections.abc import Callable
from ._qwerty_pyrt import QpuPrelude
from .pyast_utils import get_func_pyast
from .convert_ast import convert_qpu_prelude_ast
from .err import EXCLUDE_ME_FROM_STACK_TRACE_PLEASE, \
                 _cook_programmer_traceback

class PreludeHandle:
    """
    An instance of this class is returned by ``@qpu_prelude`` so that
    programmers can pass preludes to the ``@qpu`` decorator.
    """

    def __init__(self, prelude: QpuPrelude):
        self._prelude = prelude

@_cook_programmer_traceback
def qpu_prelude(func: Callable[..., ...]) -> PreludeHandle:
    """
    The ``@qpu_prelude`` decorator, which allows programmers to define their
    own ``@qpu`` prelude. Please see ``default_qpu_prelude.py`` for an example.
    """
    filename, line_offset, col_offset, func_ast = get_func_pyast(func)
    prelude = convert_qpu_prelude_ast(func_ast, filename, line_offset,
                                      col_offset)
    return PreludeHandle(prelude)
