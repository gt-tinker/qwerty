import os
import re
import glob
import shlex
import unittest
import subprocess

should_skip = bool(os.environ.get('SKIP_FILECHECK_TESTS'))

# Imitates LLVM lit[2] by dynamically building a TestCase class[1]
# [1]: https://stackoverflow.com/a/25860118/321301
# [2]: https://llvm.org/docs/CommandGuide/lit.html
def discover_filecheck_tests(cls):
    whereami = os.path.dirname(__file__)
    test_filenames = glob.glob("**/*.mlir", root_dir=whereami,
                               recursive=True)
    assert test_filenames, "No .mlir files found to test, something is broken"

    for test_filename in test_filenames:
        test_name = 'test_' + re.sub(r'[\\/-]', '_',
                                     test_filename.removesuffix('.mlir'))
        rel_filename = os.path.join(whereami, test_filename)
        with open(rel_filename) as fp:
            first_line = next(iter(fp))

        RUN_PREFIX = 'RUN: '
        run_idx = first_line.find(RUN_PREFIX)
        if run_idx < 0:
            raise ValueError('{} is missing "RUN:" line'
                             .format(test_filename))
        run_cmd_fmt = first_line[run_idx+len(RUN_PREFIX):]
        run_cmd_fmt_splat = shlex.split(run_cmd_fmt)
        run_cmd_splat = [rel_filename if tok == '%s' else tok
                         for tok in run_cmd_fmt_splat]
        pipe_indices = [-1] \
                       + [i for i, tok in enumerate(run_cmd_splat)
                          if tok == '|'] \
                       + [len(run_cmd_splat)]
        pipeline = [run_cmd_splat[l_pipe_idx+1:r_pipe_idx]
                    for l_pipe_idx, r_pipe_idx in
                        zip(pipe_indices, pipe_indices[1:])]
        if not pipeline:
            raise ValueError('Empty command in {test_filename}')

        # Use default arguments to force Python to use the current value of
        # pipeline. See:
        # 1. https://stackoverflow.com/a/54289183/321301
        # 2. https://discuss.python.org/t/make-lambdas-proper-closures/10553/3
        def test_func(self, pipeline=pipeline):
            if len(pipeline) == 1:
                cmd, = pipeline
                subprocess.run(cmd, check=True)
            else:
                first_process = subprocess.Popen(pipeline[0],
                                                 stdout=subprocess.PIPE)
                processes = [first_process]
                prev_stdout = first_process.stdout

                for cmd in pipeline[1:-1]:
                    next_process = subprocess.Popen(cmd, stdin=prev_stdout,
                                                    stdout=subprocess.PIPE)
                    processes.append(next_process)
                    prev_stdout = next_process.stdout

                last_process = subprocess.Popen(pipeline[-1],
                                                stdin=prev_stdout)
                processes.append(last_process)

                # Per a warning in the documentation[1], if you use
                # stdout=PIPE, you should call communicate() instead of
                # wait() to avoid a deadlock. (What is the deadlock? Imagine
                # the upstream process blocking on a write() while we have our
                # finger in our nose wait()ing on it [2].) It is tempting to
                # call .communicate() for each command in the pipeline
                # (left-to-right), but testing shows that p.communicate() is
                # obnoxiously calling p.stdout.read() itself, which corrupts
                # the output for processes downstream in the pipeline. Unless,
                # that is, we work backwards (right-to-left), letting consumers
                # exit and then gobbling up any remaining stdout before closing
                # stdout. Our friend .communicate() is a convenient way to do
                # the last two steps.
                # [1]: https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
                # [2]: https://stackoverflow.com/a/49728599/321301
                exit_codes = []
                for process in reversed(processes):
                    process.communicate()
                    exit_codes.append(process.returncode)

                # Doing this separately from the loop above so we do not miss
                # calling communicate() to wait for some process to exit
                for exit_code in exit_codes:
                    self.assertEqual(0, exit_code, "Process failed (see output above)")

        setattr(cls, test_name, test_func)

    return cls

@unittest.skipIf(should_skip, "Skipping FileCheck tests as requested by $SKIP_FILECHECK_TESTS")
@discover_filecheck_tests
class FileCheckTests(unittest.TestCase):
    pass
