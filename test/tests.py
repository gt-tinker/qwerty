#!/usr/bin/env python3

import os
import re
import sys
import glob
import shlex
import random
import unittest
import subprocess
from fractions import Fraction
from qwerty import *

class IntegrationTests(unittest.TestCase):
    # helper function; generates random bit[n]
    def randbits(self, n):
        random.seed(420)
        return bit[n](random.randrange(0, 1 << n))

    def test_bell(self):
        from integration_tests import bell, bell_p
        output = str(bell.test())
        self.assertIn(output, ['00', '11'], "Final output was not the 1st Bell State!")

        output = str(bell_p.test())
        self.assertIn(output, ['00', '11'], "Final output was not the 1st Bell State!")

    def test_dj(self):
        from integration_tests import dj
        output = str(dj.test())
        self.assertEqual(output, "f(x) is constant | f(x) is balanced", "Final outputs were not constant and balanced!")

    def test_bv(self):
        from integration_tests import bv
        secret_bits = self.randbits(n=4)
        secret_str = str(secret_bits)
        output = str(bv.bv(secret_bits))
        self.assertEqual(output, secret_str, "Secret String {} was not found!".format(secret_str))

    def test_assignment(self):
        from integration_tests import assignment
        expected_value = bit[1](0)
        actual = assignment.test()
        self.assertEqual(expected_value, actual, f"Assignmnent with hinting: Expected a 0, got {actual}")

    def test_shors(self):
        from integration_tests import shors
        output = None
        for i in range(32):
            try:
                output = str(shors.test(15)) # currently set to 15 for testing
            except ValueError as e:
                print(str(e) + ' (this is expected)... ', file=sys.stderr, end='', flush=True)
            else:
                break
        else:
            self.assertTrue(False, "shors never produced the right answer")
        self.assertIn(output, ['3', '5'], "Shor's Failed!")

    def test_simons(self):
        from integration_tests import simons
        output = str(simons.test())
        self.assertEqual(output, "101", "Wrong secret string")

    def test_64plus_qubits(self):
        from integration_tests import apint
        output = str(apint.test())
        self.assertEqual(output, "101010101010101010101010101010101010101010101010101010101010101011")

    def test_conditional(self):
        from integration_tests import cond
        output = "1"
        for i in range(128):
            output = str(cond.test())
            if output != "1":
                break
        else:
            self.assertTrue(False, "cond took too long to produce the right output")
        self.assertEqual(output, "0")

    def test_adjoint(self):
        from integration_tests import adjoint
        n_shots = 1024
        histo_expected = {bit[1](0b0): n_shots}
        self.assertEqual(histo_expected, adjoint.test(n_shots))
    
    def test_grovers(self):
        from integration_tests import grovers
        ans2 = grovers.test(2)
        ans3 = grovers.test(3)
        for answer in ans2:
            self.assertEqual(str(answer), '11')
        for answer in ans3:
            self.assertEqual(str(answer), '111')

    def test_teleport(self):
        from integration_tests import teleport
        n_shots = 1024
        histo_expected = {bit[1](0b1): n_shots}
        self.assertEqual(histo_expected, teleport.test(n_shots))

    def test_ghz5(self):
        from integration_tests import ghz
        histo = ghz.test(5)
        histo_keys_expected = {bit[5](0b11111), bit[5](0b00000)}
        self.assertEqual(histo_keys_expected, set(histo.keys()))

    def test_bitensor_unit(self):
        from integration_tests import bitensor_unit
        n_shots = 1024
        histo = bitensor_unit.test(n_shots)
        histo_expected = {bit[1](0b1): n_shots}
        self.assertEqual(histo_expected, histo)

    def test_swap_inference(self):
        from integration_tests import swap_inference
        n_shots = 1024
        cases = [(bit[7](0b101_1001), bit[7](0b101_1101)),
                 (bit[7](0b010_1001), bit[7](0b010_1101)),
                 (bit[7](0b110_1001), bit[7](0b110_1001))]

        for input_, output in cases:
            histo = swap_inference.test(input_, n_shots)
            histo_expected = {output: n_shots}
            self.assertEqual(histo_expected, histo)

class RuntimeTests(unittest.TestCase):
    def test_bit_str(self):
        b = bit[4](0b1101)
        self.assertEqual(str(b), '1101')

    def test_bit_str_leading_zeros(self):
        b = bit[8](0b00001101)
        self.assertEqual(str(b), '00001101')

    def test_bit_repr(self):
        b = bit[4](0b1101)
        self.assertEqual(repr(b), 'qwerty.bit[4](0b1101)')

    def test_bit_repr_leading_zeros(self):
        b = bit[8](0b00001101)
        self.assertEqual(repr(b), 'qwerty.bit[8](0b00001101)')

    def test_bit_nonint_n_bits(self):
        with self.assertRaisesRegex(TypeError, "int.*, not float"):
            bit[4.0](0b1101)

    def test_bit_nonint_val(self):
        with self.assertRaisesRegex(TypeError, "int.*, not float"):
            bit[4](13.0)

    def test_bit_zero_n_bits(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            bit[0](0b0)

    def test_bit_neg_n_bits(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            bit[-1](0b0)

    def test_cfrac_from_fraction_nonmangle_input(self):
        frac = Fraction(2, 3) # 2/3
        cf = cfrac.from_fraction(frac)
        self.assertEqual(frac, Fraction(2, 3))

    def test_cfrac_from_fraction_negative(self):
        frac = Fraction(-1, 3) # -1/3
        with self.assertRaisesRegex(NotImplementedError, "negative .* not supported"):
            cf = cfrac.from_fraction(frac)

    def test_cfrac_from_fraction_zero(self):
        frac = Fraction(0)
        cf = cfrac.from_fraction(frac)
        self.assertEqual(cf.partial_denoms, [0])

    def test_cfrac_from_fraction_trivial(self):
        frac = Fraction(3, 2) # 1 + 1/2
        cf = cfrac.from_fraction(frac)
        self.assertEqual(cf.partial_denoms, [1, 2])

    def test_cfrac_from_fraction_gt_1(self):
        frac = Fraction(62, 23) # 62/23
        cf = cfrac.from_fraction(frac)
        # Section 12.2 of Rosen's "Elementary Number Theory" 6th ed.
        self.assertEqual(cf.partial_denoms, [2, 1, 2, 3, 2])

    def test_cfrac_from_fraction_lt_1(self):
        frac = Fraction(13, 31) # 13/31
        # 13/31 = 0 + 1/(31/13)
        # 31/13 = [2, 2, 1, 1, 2] by Box 5.3 in Nielsen and Chuang
        # thus 13/31 = [0; 2, 2, 1, 1, 2]
        cf = cfrac.from_fraction(frac)
        self.assertEqual(cf.partial_denoms, [0, 2, 2, 1, 1, 2])

    def test_cfrac_convergents_trivial(self):
        frac = Fraction(3, 2) # 1 + 1/2
        cf = cfrac.from_fraction(frac)
        self.assertEqual(cf.partial_denoms, [1, 2])
        self.assertEqual(cf.convergents(), [
            Fraction(1, 1), # 1/1
            Fraction(3, 2), # 3/2
        ])

    def test_cfrac_convergents_nontrivial(self):
        frac = Fraction(173, 55) # 173/55
        cf = cfrac.from_fraction(frac)
        # Example 12.7 in Rosen's "Elementary Number Theory" 6th ed.
        self.assertEqual(cf.partial_denoms, [3, 6, 1, 7])
        self.assertEqual(cf.convergents(), [
            Fraction(3, 1), # 3/1
            Fraction(19, 6), # 19/6
            Fraction(22, 7), # 22/7
            Fraction(173, 55), # 173/55
        ])

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

@discover_filecheck_tests
class FileCheckTests(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
