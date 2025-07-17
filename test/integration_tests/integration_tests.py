import os
import sys
import unittest
from qwerty import *

should_skip = bool(os.environ.get('SKIP_INTEGRATION_TESTS'))

@unittest.skipIf(should_skip, "Skipping integration tests as requested by $SKIP_INTEGRATION_TESTS")
class IntegrationTests(unittest.TestCase):
    def test_bell(self):
        from .tests import bell, bell_p
        output = str(bell.test())
        self.assertIn(output, ['00', '11'], "Final output was not the 1st Bell State!")

        output = str(bell_p.test())
        self.assertIn(output, ['00', '11'], "Final output was not the 1st Bell State!")

    def test_dj(self):
        from .tests import dj
        output = str(dj.test())
        self.assertEqual(output, "f(x) is constant | f(x) is balanced", "Final outputs were not constant and balanced!")

    def test_bv(self):
        from .tests import bv
        secret_bits = bit[4](0b1101)
        output = bv.bv(secret_bits)
        self.assertEqual(output, secret_bits, "Secret String {} was not found!".format(secret_bits))

    def test_assignment(self):
        from .tests import assignment
        expected_value = bit[1](0)
        actual = assignment.test()
        self.assertEqual(expected_value, actual, f"Assignmnent with hinting: Expected a 0, got {actual}")

    def test_shors(self):
        from .tests import shors
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
        from .tests import simons
        output = str(simons.test())
        self.assertEqual(output, "101", "Wrong secret string")

    def test_64plus_qubits(self):
        from .tests import apint
        output = str(apint.test())
        self.assertEqual(output, "101010101010101010101010101010101010101010101010101010101010101011")

    def test_conditional(self):
        from .tests import cond
        output = "1"
        for i in range(128):
            output = str(cond.test())
            if output != "1":
                break
        else:
            self.assertTrue(False, "cond took too long to produce the right output")
        self.assertEqual(output, "0")

    def test_adjoint(self):
        from .tests import adjoint
        n_shots = 1024
        histo_expected = {bit[1](0b0): n_shots}
        self.assertEqual(histo_expected, adjoint.test(n_shots))

    def test_grovers(self):
        from .tests import grovers
        ans2 = grovers.test(2)
        ans3 = grovers.test(3)
        for answer in ans2:
            self.assertEqual(str(answer), '11')
        for answer in ans3:
            self.assertEqual(str(answer), '111')

    def test_teleport(self):
        from .tests import teleport
        n_shots = 1024
        histo_expected = {bit[1](0b1): n_shots}
        self.assertEqual(histo_expected, teleport.test(n_shots))

    def test_ghz5(self):
        from .tests import ghz
        histo = ghz.test(5)
        histo_keys_expected = {bit[5](0b11111), bit[5](0b00000)}
        self.assertEqual(histo_keys_expected, set(histo.keys()))

    def test_bitensor_unit(self):
        from .tests import bitensor_unit
        n_shots = 1024
        histo = bitensor_unit.test(n_shots)
        histo_expected = {bit[1](0b1): n_shots}
        self.assertEqual(histo_expected, histo)

    def test_swap_inference(self):
        from .tests import swap_inference
        n_shots = 1024
        cases = [(bit[7](0b101_1001), bit[7](0b101_1101)),
                 (bit[7](0b010_1001), bit[7](0b010_1101)),
                 (bit[7](0b110_1001), bit[7](0b110_1001))]

        for input_, output in cases:
            histo = swap_inference.test(input_, n_shots)
            histo_expected = {output: n_shots}
            self.assertEqual(histo_expected, histo)

    def test_syntax_sugar(self):
        from .tests import basis_sugar
        n_shots = 1024
        histo = basis_sugar.sweet(shots=n_shots, histogram=True)
        histo_expected = {bit[1](0b0): n_shots}
        self.assertEqual(histo, histo_expected)
    
    def test_arith_select(self):
        from .tests import arith_select
        output = arith_select.test()
        self.assertIn(output, [bit[1](0b0), bit[1](0b1)], "Final output was not in the select options!")
