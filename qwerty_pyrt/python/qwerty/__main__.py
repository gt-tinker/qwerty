"""
Run a Qwerty REPL when users say::

    python -m qwerty repl
"""

import sys
from pathlib import Path
from .repl import repl, get_jupyter_kernel

def run_repl():
    # Give users a nicer prompt with basic history if it's available on their OS
    try:
        import readline
        using_readline = True
    except ModuleNotFoundError:
        using_readline = False
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

    return exit_code

def run_jupyter(more_args):
    from ipykernel.kernelapp import IPKernelApp
    kernel_class = get_jupyter_kernel()
    IPKernelApp.launch_instance(kernel_class=kernel_class, argv=more_args)
    # Not reachable
    return 0

def main(argv):
    args = argv[1:]
    if len(args) >= 1:
        cmd = args[0]
        more_args = args[1:]
        if cmd == 'repl':
            if more_args:
                print('repl does not take any arguments', file=sys.stderr)
            else:
                return run_repl()
        elif cmd == 'jupyter':
            return run_jupyter(more_args)
        else:
            print(f'unknown subcommand {cmd}', file=sys.stderr)

    print('usage: python -m qwerty repl     Start a Qwerty REPL\n'
          '       python -m qwerty jupyter  Start a Qwerty Jupyter kernel', file=sys.stderr)
    return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))
