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

    def test_bv_nometa(self):
        from .integ import bv_nometa
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_nometa.test(shots))

    @unittest.skip('Superdense coding still WIP')
    def test_superdense_nocap(self):
        from .integ import superdense_nocap
        shots = 1024
        expected_histo = {bit[2](0b00): shots}
        self.assertEqual(expected_histo, superdense_nocap.test(shots))
