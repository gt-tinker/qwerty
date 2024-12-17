"""
This is the Python runtime for the Qwerty programming language.

You should import the Qwerty runtime at the beginning of your .py file as
follows::

    from qwerty import *

This is necessary to use Qwerty syntax, e.g., type annotations on @qpu kernels.
"""

import string
import warnings
from .jit import *
from .runtime import *

# Disable warnings for Qwerty syntax like {'0','1'}[5]
# TODO: Allow consumers of the qwerty module to configure this instead of
#       unilaterally deciding this on their behalf
warnings.simplefilter('ignore', category=SyntaxWarning)

_all_jit = ['qpu', 'classical', 'dump_mlir_module', 'dump_qir', 'get_qir',
            'set_func_opt']
_all_types = ['bit', 'qfunc', 'cfunc', 'rev_qfunc', 'rev_func', 'dimvar',
              'qubit', 'func', 'cfrac', 'angle', 'reversible',
              'print_histogram'] + list(string.ascii_uppercase)

__all__ = _all_jit + _all_types
