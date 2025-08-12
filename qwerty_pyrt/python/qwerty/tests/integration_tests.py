import os
import unittest
from qwerty.runtime import bit
from qwerty.kernel import _reset_compiler_state

should_skip = bool(os.environ.get('SKIP_INTEGRATION_TESTS'))

@unittest.skipIf(should_skip, "Skipping integration tests as requested by $SKIP_INTEGRATION_TESTS")
class NoMetaIntegrationTests(unittest.TestCase):
    def setUp(self):
        _reset_compiler_state()

    def test_randbit(self):
        from .integ.nometa import randbit
        shots = 1024
        actual_histo = randbit.test(shots)
        zero, one = bit[1](0b0), bit[1](0b1)
        self.assertGreater(actual_histo.get(zero, 0), shots//8, "Too few zeros")
        self.assertGreater(actual_histo.get(one, 0), shots//8, "Too few ones")
        self.assertEqual(shots, actual_histo.get(zero, 0) + actual_histo.get(one, 0), "missing shots")

    def test_interproc(self):
        # Like randbit above except involves a call from one kernel to another
        from .integ.nometa import interproc
        shots = 1024
        actual_histo = interproc.test(shots)
        zero, one = bit[1](0b0), bit[1](0b1)
        self.assertGreater(actual_histo.get(zero, 0), shots//8, "Too few zeros")
        self.assertGreater(actual_histo.get(one, 0), shots//8, "Too few ones")
        self.assertEqual(shots, actual_histo.get(zero, 0) + actual_histo.get(one, 0), "missing shots")

    def test_baby_classical(self):
        from .integ.nometa import baby_classical
        shots = 1024
        expected_histo = {bit[3](0b111): shots}
        self.assertEqual(expected_histo, baby_classical.test(shots))

    def test_bv_noclassical(self):
        from .integ.nometa import bv_noclassical
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_noclassical.test(shots))

    def test_bv(self):
        from .integ.nometa import bv_nocap
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_nocap.test(shots))

    def test_bv(self):
        from .integ.nometa import bv
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv.test(shots))

    def test_func_tens(self):
        from .integ.nometa import func_tens
        shots = 1024
        expected_histo = {bit[1](0b1): shots}
        self.assertEqual(expected_histo, func_tens.test(shots))

    def test_pack_unpack(self):
        from .integ.nometa import pack_unpack
        shots = 1024
        expected_histo = {bit[3](0b101): shots}
        self.assertEqual(expected_histo, pack_unpack.test(shots))

    def test_pad(self):
        from .integ.nometa import pad
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, pad.test(shots))

    def test_superdense_nocap(self):
        from .integ.nometa import superdense_nocap
        shots = 1024
        expected_histos = (
            {bit[2](0b00): shots},
            {bit[2](0b01): shots},
            {bit[2](0b10): shots},
            {bit[2](0b11): shots},
        )
        self.assertEqual(expected_histos, superdense_nocap.test(shots))

    def test_teleport(self):
        from .integ.nometa import teleport
        shots = 1024
        expected_histos = (
            {bit[1](0b0): shots},
            {bit[1](0b1): shots},
            {bit[1](0b0): shots},
            {bit[1](0b1): shots},
        )
        self.assertEqual(expected_histos, teleport.test(shots))

    def test_tilt(self):
        from .integ.nometa import tilt
        shots = 1024
        expected_histos = (
            {bit[1](0b0): shots},
            {bit[1](0b1): shots},
            {bit[1](0b0): shots},
            {bit[1](0b1): shots},
        )
        self.assertEqual(expected_histos, tilt.test(shots))

    def test_fourier(self):
        from .integ.nometa import fourier
        shots = 1024
        expected_histo = {bit[3](0b101): shots}
        self.assertEqual(expected_histo, fourier.test(shots))

@unittest.skipIf(should_skip, "Skipping integration tests as requested by $SKIP_INTEGRATION_TESTS")
class MetaNoInferIntegrationTests(unittest.TestCase):
    def setUp(self):
        _reset_compiler_state()

    def test_bv_nomacro_noclassical(self):
        from .integ.meta_noinfer import bv_nomacro_noclassical
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_nomacro_noclassical.test(shots))

    def test_bv_somemacro_noclassical(self):
        from .integ.meta_noinfer import bv_somemacro_noclassical
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_somemacro_noclassical.test(shots))

    def test_bv_macro_noclassical(self):
        from .integ.meta_noinfer import bv_macro_noclassical
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_macro_noclassical.test(shots))

    def test_bv_macro_classical(self):
        from .integ.meta_noinfer import bv_macro_classical
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_macro_classical.test(shots))

    def test_fourier(self):
        from .integ.meta_noinfer import fourier
        shots = 1024
        expected_histo = {bit[3](0b101): shots}
        self.assertEqual(expected_histo, fourier.test(shots))
