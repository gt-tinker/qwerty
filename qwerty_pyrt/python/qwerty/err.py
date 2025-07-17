"""
An elaborate trick for rewriting Python tracebacks so that Qwerty syntax/type
errors point to the offending line of the programmer's code rather than the
point in the runtime where the error was thrown.

For technical details on how this trick works, see the lengthy comment in the
``err.py`` source code. You can't miss it.
"""

import sys
from types import FrameType, TracebackType
from typing import Optional

# Used to exclude code in this file (or a file that imports this) from showing
# up in compiler error tracebacks. See _cook_programmer_traceback() below
EXCLUDE_ME_FROM_STACK_TRACE_PLEASE = 1

# Mapping of DebugLocs to Frames
_FRAME_MAP = {}

class QwertyProgrammerError(Exception):
    """
    An error triggered by a programmer mistake and not a compiler author
    skill issue. Instances of this exception will be caught by
    ``@_cook_programmer_traceback`` below.
    """

    def __init__(self, msg, dbg=None):
        """
        Create a ``QwertySyntaxError`` with the given error message. When the
        keyword argument ``dbg==None``, that means "when ``err.py`` intercepts
        this exception, the first frame not in the Qwerty runtime is actually
        what we want — no need to mess around with the line number."
        """
        self.dbg = dbg
        # _qwerty_harness.cpp does this in qwerty_programmer_error() as well.
        # The reason is that our stack trace hack is not able to draw a little
        # arrow to point at the offending column
        super().__init__(msg + " (at column " + str(col) + ")"
                         if dbg is not None and (col := dbg.get_col()) > 0
                         else msg)

    def kind(self) -> str:
        """Return a user-friendly description of what time of error this is."""
        return 'Error'

class QwertyTypeError(QwertyProgrammerError):
    """
    A typing problem (practically, a problem encountered during the Rust type
    checking code).
    """
    def __init__(self, msg, dbg=None):
        super().__init__(msg, dbg)

class QwertySyntaxError(QwertyProgrammerError):
    """
    A syntax mistake or unsupported Python syntax (practically, a problem
    encountered during the Python convert_ast() code).
    """
    def __init__(self, msg, dbg=None):
        super().__init__(msg, dbg)

    def kind(self) -> str:
        return 'Syntax Error'

def set_dbg_frame(dbg, frame):
    global _FRAME_MAP

    if dbg is not None and frame is not None and dbg not in _FRAME_MAP:
        _FRAME_MAP[dbg] = frame

def get_frame() -> Optional[FrameType]:
    """
    Used to capture the stack frame at the time ``convert_ast()`` is called,
    i.e., when ``@qpu`` is initially called. See
    ``@_cook_programmer_traceback`` below for details.
    """
    # Try to have less of a hard dependence on CPython (copium)
    if hasattr(sys, '_getframe'):
        return sys._getframe()
    else:
        return None

def _strip_runtime_frames(frame):
    """
    Trick to pop compiler/runtime stack frames off the stack trace ("frame") we
    use to cook up a fake backtrace into the programmer's code. See
    ``@_cook_programmer_traceback`` below for details.
    """
    while 'EXCLUDE_ME_FROM_STACK_TRACE_PLEASE' in frame.f_globals \
          and frame.f_back is not None:
        frame = frame.f_back
    return frame

# Before implementing this, here was the lifecycle of a Qwerty type checker
# error:
#
# 1. Thrown in C++ code as a C++ exception (TypeException) with DebugInfo
#    attached
# 2. Caught in _qwerty_harness, and rethrown as a Python exception
#
# That is, Qwerty compiler errors bubbled up all the way from _qwerty_harness
# to the programmer. The result looked something like this:
#
#     $ python3 example.py
#     Traceback (most recent call last):
#       File "example.py", line 10, in <module>
#         gigatroll()
#       File "example.py", line 4, in gigatroll
#         @qpu
#          ^^^
#       File "[snip]/venv/lib/python3.11/site-packages/qwerty/jit.py", line 619, in __call__
#         return self._proxy_to(func)
#                ^^^^^^^^^^^^^^^^^^^^
#       File "[snip]/venv/lib/python3.11/site-packages/qwerty/jit.py", line 625, in _proxy_to
#         return _jit(AST_QPU, func, captures, self._last_dimvars)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#       File "[snip]/venv/lib/python3.11/site-packages/qwerty/jit.py", line 553, in _jit
#         qwerty_ast.typecheck()
#     qwerty.QwertyTypeError: Input to pipe Qubit does not match right-hand input type Qubit[2] at line 5, column 24 in example.py
#
# The majority of that stack trace is not useful. Why is code inside the Qwerty
# compiler/runtime being printed instead of, I don't know, _the offending
# Qwerty code?_
#
# Producing a more helpful error is not easy, though. It's tempting to bypass
# the Python exception mechanism entirely, but I don't want to unexpectedly
# sys.exit() inside a user's program. And merely printfing an error message to
# stderr does not correctly disrupt the caller's Python control flow. So okay,
# we need to throw a Python exception.
#
# Okay, so just throw an exception with a custom backtrace, right? Easy! No,
# it's not so easy. A Python traceback is built with frames, which themselves
# cannot be constructed from Python or the Stable ABI. Worse still, even if you
# could cook up a traceback just how you liked, when you raise an exception
# with that traceback from either Python or C, the interpreter _modifies_ your
# traceback as the exception travels upwards in the execution stack.
# Specifically, it prepends more frames, so as the exception would trickle
# upward from the depths of the Qwerty runtime back to user code, it would
# collect all those annoying frames such as _proxy_to() and _jit() in the
# example above along the way.
#
# The solution used here is based on the following insights:
#
# 1. Perform traceback rewriting right at the entry point methods into the
#    runtime to avoid the annoying "bread crumb" frames as the exception
#    propagates after we set its traceback.
#    (To avoid repeating this code for every user-facing method that could
#     eventually call into _qwerty_harness code that raises an exception, we
#     put the traceback rewriting code in a decorator
#     @_cook_programmer_traceback)
# 2. Modify the traceback of the exception while it is already in-flight.
#    (We do this below by catching the exception and re-raising it, but
#     carefully modifying the in-flight exception using a finally block.)
# 3. Use the tb_lineno field of tracebacks, which we can write to, rather than
#    trying to hack a line number into a frame.
#    The following line in the documentation is our saving grace:
#    > The line number and last instruction in the traceback may differ from
#    > the line number of its frame object if the exception occurred in a try
#    > statement with no matching except clause or with a finally clause.
#    (That's not the situation happening here, but we can abuse the
#     functionality nonetheless.)
#
# But we still need _some_ frame to put in the traceback, and in fact, that
# frame needs to be from the user's source file for the line number hack
# (insight #3) to work. The trick we use is simply storing the current stack
# frame at AST parse time inside DebugInfo, which is already propagated for
# errors.
#
# So now the process works as follows (new steps are marked with ***):
#
# *** 0. In convert_ast, sys._getframe() is stored in every DebugInfo
#     1. Thrown in C++ code as a C++ exception (TypeException) with DebugInfo
#        attached
#     2. Caught in _qwerty_harness, and rethrown as a Python exception
# *** 3. Error propagates with ugly traceback
# *** 4. At the last possible moment before the exception escapes the Qwerty
#        runtime (aka in @_cook_programmer_traceback), we replace the
#        exception's traceback with a new traceback. Our new traceback has the
#        desired line number and sets the frame to the frame stored in the
#        DebugInfo with every frame with our sentinel symbol
#        EXCLUDE_ME_FROM_STACK_TRACE_PLEASE as a global variable popped off.
#
# Here's the result:
#
#     $ python3 example.py
#     Traceback (most recent call last):
#       File "example.py", line 10, in <module>
#         gigatroll()
#       File "example.py", line 4, in gigatroll
#         @qpu
#          ^^^
#       File "example.py", line 6, in gigatroll
#         return '1' | {'0','1'} + {'0','1'} >> {'00', '01', '11', '10'} | std[2].measure
#
#     qwerty.QwertyTypeError: Input to pipe Qubit does not match right-hand input type Qubit[2] (at column 16)
#
# Much better! Why aren't portions of the line highlighted, though?
# Unfortunately, I could not figure out how to do that. There is a tb_lasti
# field of tracebacks, which the documentation unhelpfully defines as
# "Indicates the 'precise instruction'." In CPython, tb_lasti is calculated with some
# pointer arithmetic that appears to calculate a bytecode offset (check out the
# implementation of PyTraceBack_Here()). Unfortunately, __code__.co_positions()
# and __code__.co_lines() appear to be just barely unsuitable for calculating
# this offset. Even if they were capable, though, one must consider that not
# all syntax may show up in the bytecode. Since we are cooking an exception
# here rather than one organically thrown by bytecode running, it possible we
# are raising exceptions for syntactic constructs that never become bytecode.
# So tl;dr we pass -1 for tb_lasti, which appears to successfully prevent
# carats from being printed in confusing places at least. As consolation, we
# print the column number in the error message as shown in the example above.
def _cook_programmer_traceback(f):
    """
    An elaborate trick to rewrite Qwerty compiler error tracebacks to make them
    more programmer-friendly. This decorator should be used on _every_
    public-facing Qwerty runtime method/function that could throw a
    ``QwertyProgrammerError``. See the lengthy comment in ``err.py`` for
    details.
    """

    global _FRAME_MAP

    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except QwertyProgrammerError as e:
            dbg = e.dbg
            if dbg is not None and (frame := _FRAME_MAP.get(dbg)) is not None:
                frame = _strip_runtime_frames(frame)
                tb = TracebackType(None, frame, -1, dbg.get_line())
            else:
                # Special case: if dbg is None, that means the most recent
                # non-runtime frame in the linked list of frames is
                # actually what we want to show up. So just strip off the
                # frames without monkeying with the line number or bytecode
                # index (lasti)
                frame = _strip_runtime_frames(e.__traceback__.tb_frame)
                tb = TracebackType(None, frame, frame.f_lasti, frame.f_lineno)

            # Why call e.with_traceback(tb) here yet also set
            # e.__traceback__ below? First, some background:
            #
            # In CPython versions before 3.11, the current exception
            # state had three parts: an exception type, an exception
            # value (usually an instance of Exception), and a traceback
            # object. When you raised an exception, the interpreter took
            # your Exception object and filled these three fields
            # accordingly, including the traceback. So your only hope of
            # modifying the traceback in CPython <3.11 is changing it on
            # the Exception instance _before_ raising it.
            #
            # In CPython 3.11, however, exception tracking was changed
            # to store the traceback in the exception itself — the three
            # fields mentioned were merged into one, the Exception
            # object itself. This means if you can access the Exception
            # object e after it is thrown, you _can_ modify the
            # traceback by changing e.__traceback__.
            #
            # So taking both approaches (e.with_traceback() and
            # e.__traceback__) is a defensive move to handle both
            # CPython <3.11 and >=3.11 with the same code. In CPython
            # <3.11, the e.__traceback__ modification will do nothing,
            # and `raise e' below will show up in the traceback, but at
            # least the traceback isn't clogged with a deep nest of
            # compiler calls. In CPython >=3.11, modifying
            # e.__traceback__ clobbers the traceback after the exception
            # is thrown, removing `raise e' from the traceback and making
            # e.with_traceback() harmless.
            e = e.with_traceback(tb)

            try:
                # The comment below is for Qwerty programmers
                raise e # Ignore me
            finally:
                e.__traceback__ = tb

    return wrapper
