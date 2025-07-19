"""
Run a Qwerty REPL when users say::

    python -m qwerty
"""

# Give users a nicer prompt with basic history
try:
    import readline
except ModuleNotFoundError:
    # ...if it's available on their OS
    pass

from .repl import repl

repl(input, print)
