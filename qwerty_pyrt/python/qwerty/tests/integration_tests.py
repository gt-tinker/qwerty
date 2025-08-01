import os
import unittest
from qwerty.runtime import bit
from qwerty.kernel import _reset_compiler_state

should_skip = bool(os.environ.get('SKIP_INTEGRATION_TESTS'))

@unittest.skipIf(should_skip, "Skipping integration tests as requested by $SKIP_INTEGRATION_TESTS")
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        _reset_compiler_state()

    def test_randbit_nometa(self):
        from .integ import randbit_nometa
        shots = 1024
        actual_histo = randbit_nometa.test(shots)
        zero, one = bit[1](0b0), bit[1](0b1)
        self.assertGreater(actual_histo.get(zero, 0), shots//8, "Too few zeros")
        self.assertGreater(actual_histo.get(one, 0), shots//8, "Too few ones")
        self.assertEqual(shots, actual_histo.get(zero, 0) + actual_histo.get(one, 0), "missing shots")

    def test_interproc_nometa(self):
        # Like randbit_nometa above except involves a call from one kernel to another
        from .integ import interproc_nometa
        shots = 1024
        actual_histo = interproc_nometa.test(shots)
        zero, one = bit[1](0b0), bit[1](0b1)
        self.assertGreater(actual_histo.get(zero, 0), shots//8, "Too few zeros")
        self.assertGreater(actual_histo.get(one, 0), shots//8, "Too few ones")
        self.assertEqual(shots, actual_histo.get(zero, 0) + actual_histo.get(one, 0), "missing shots")

    def test_bv_nometa(self):
        from .integ import bv_nometa
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_nometa.test(shots))

    def test_func_tens_nometa(self):
        from .integ import func_tens_nometa
        shots = 1024
        expected_histo = {bit[1](0b1): shots}
        self.assertEqual(expected_histo, func_tens_nometa.test(shots))

    def test_pack_unpack_nometa(self):
        from .integ import pack_unpack_nometa
        shots = 1024
        expected_histo = {bit[3](0b101): shots}
        self.assertEqual(expected_histo, pack_unpack_nometa.test(shots))

    def test_pad_nometa(self):
        from .integ import pad_nometa
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, pad_nometa.test(shots))

    def test_superdense_nocap(self):
        from .integ import superdense_nocap
        shots = 1024
        expected_histos = (
            {bit[2](0b00): shots},
            {bit[2](0b01): shots},
            {bit[2](0b10): shots},
            {bit[2](0b11): shots},
        )
        self.assertEqual(expected_histos, superdense_nocap.test(shots))

    def test_teleport_nometa(self):
        from .integ import teleport_nometa
        shots = 1024
        expected_histos = (
            {bit[1](0b0): shots},
            {bit[1](0b1): shots},
            {bit[1](0b0): shots},
            {bit[1](0b1): shots},
        )
        self.assertEqual(expected_histos, teleport_nometa.test(shots))
