"""
Run a Qwerty REPL when users say::

    python -m qwerty
"""

# Give users a nicer prompt with basic history if it's available on their OS
try:
    import readline
    using_readline = True
except ModuleNotFoundError:
    using_readline = False

import sys
from pathlib import Path
from .repl import repl

# Load the readline history if present
if using_readline:
    try:
        homedir = Path.home()
    except RuntimeError:
        # No home directory. Wow.
        hist_path = None
    else:
        hist_path = str(homedir / '.qwerty_history')
        try:
            readline.read_history_file(hist_path)
        except FileNotFoundError:
            # No problem. We'll create it later.
            pass
        except OSError:
            # A permissions problem or something. Let's not bother.
            hist_path = None
else:
    hist_path = None

try:
    exit_code = repl(input, print)
finally:
    if hist_path is not None:
        try:
            readline.write_history_file(hist_path)
        except OSError:
            # Possible permissions problem. No hard feelings.
            pass

sys.exit(exit_code)
