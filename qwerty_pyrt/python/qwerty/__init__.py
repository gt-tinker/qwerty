"""
This is the Python runtime for the Qwerty programming language.

You should import the Qwerty runtime at the beginning of your .py file as
follows::

    from qwerty import *

This is necessary to use Qwerty syntax, e.g., type annotations on @qpu kernels.
"""

import string
from .kernel import qpu, classical
from .runtime import *
from .prelude import qpu_prelude
from .default_qpu_prelude import default_qpu_prelude

_all_kernel = ['qpu', 'classical']
_all_prelude = ['qpu_prelude']
_all_runtime = ['bit', 'qfunc', 'cfunc', 'rev_qfunc', 'rev_func', 'dimvar',
                'qubit', 'func', 'cfrac', 'reversible', 'histogram'] + list(string.ascii_uppercase)

__all__ = _all_kernel + _all_prelude + _all_runtime
