import os
import unittest
from qwerty.runtime import bit
from qwerty.kernel import _reset_compiler_state
from qwerty.err import QwertyTypeError

should_skip = bool(os.environ.get('SKIP_INTEGRATION_TESTS'))
skip_msg = "Skipping integration tests as requested by $SKIP_INTEGRATION_TESTS"

@unittest.skipIf(should_skip, skip_msg)
class NoMetaIntegrationTests(unittest.TestCase):
    """Integration tests that do not use metaQwerty features at all."""

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

    def test_bv_nocap(self):
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

@unittest.skipIf(should_skip, skip_msg)
class MetaNoPreludeNoInferIntegrationTests(unittest.TestCase):
    """
    Integration tests that do use metaQwerty features but have the default
    prelude disabled.
    """

    def setUp(self):
        _reset_compiler_state()

    def test_bv_nomacro_noclassical(self):
        from .integ.meta_noprelude_noinfer import bv_nomacro_noclassical
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_nomacro_noclassical.test(shots))

    def test_bv_somemacro_noclassical(self):
        from .integ.meta_noprelude_noinfer import bv_somemacro_noclassical
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_somemacro_noclassical.test(shots))

    def test_bv_macro_noclassical(self):
        from .integ.meta_noprelude_noinfer import bv_macro_noclassical
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_macro_noclassical.test(shots))

    def test_bv_macro_classical(self):
        from .integ.meta_noprelude_noinfer import bv_macro_classical
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_macro_classical.test(shots))

    def test_fourier(self):
        from .integ.meta_noprelude_noinfer import fourier
        shots = 1024
        expected_histo = {bit[3](0b101): shots}
        self.assertEqual(expected_histo, fourier.test(shots))

    def test_custom_prelude(self):
        from .integ.meta_noprelude_noinfer import custom_prelude
        shots = 1024
        expected_histos = ({bit[1](0b0): shots},
                           {bit[1](0b1): shots},
                           {bit[1](0b0): shots},
                           {bit[1](0b1): shots},
                           {bit[1](0b0): shots},
                           {bit[1](0b1): shots})
        self.assertEqual(expected_histos, custom_prelude.test(shots))

@unittest.skipIf(should_skip, skip_msg)
class MetaNoInferIntegrationTests(unittest.TestCase):
    """
    Integration tests that use full metaQwerty features but do not rely on type
    inference.
    """

    def setUp(self):
        _reset_compiler_state()

    def test_randbit(self):
        from .integ.meta_noinfer import randbit
        shots = 1024
        actual_histo = randbit.test(shots)
        zero, one = bit[1](0b0), bit[1](0b1)
        self.assertGreater(actual_histo.get(zero, 0), shots//8, "Too few zeros")
        self.assertGreater(actual_histo.get(one, 0), shots//8, "Too few ones")
        self.assertEqual(shots, actual_histo.get(zero, 0) + actual_histo.get(one, 0), "missing shots")

    def test_interproc(self):
        # Like randbit above except involves a call from one kernel to another
        from .integ.meta_noinfer import interproc
        shots = 1024
        actual_histo = interproc.test(shots)
        zero, one = bit[1](0b0), bit[1](0b1)
        self.assertGreater(actual_histo.get(zero, 0), shots//8, "Too few zeros")
        self.assertGreater(actual_histo.get(one, 0), shots//8, "Too few ones")
        self.assertEqual(shots, actual_histo.get(zero, 0) + actual_histo.get(one, 0), "missing shots")

    def test_baby_classical(self):
        from .integ.meta_noinfer import baby_classical
        shots = 1024
        expected_histo = {bit[3](0b111): shots}
        self.assertEqual(expected_histo, baby_classical.test(shots))

    def test_bv_noclassical(self):
        from .integ.meta_noinfer import bv_noclassical
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_noclassical.test(shots))

    def test_bv_nocap(self):
        from .integ.meta_noinfer import bv_nocap
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv_nocap.test(shots))

    def test_bv(self):
        from .integ.meta_noinfer import bv
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, bv.test(shots))

    def test_func_tens(self):
        from .integ.meta_noinfer import func_tens
        shots = 1024
        expected_histo = {bit[1](0b1): shots}
        self.assertEqual(expected_histo, func_tens.test(shots))

    def test_pack_unpack(self):
        from .integ.meta_noinfer import pack_unpack
        shots = 1024
        expected_histo = {bit[3](0b101): shots}
        self.assertEqual(expected_histo, pack_unpack.test(shots))

    def test_pad(self):
        from .integ.meta_noinfer import pad
        shots = 1024
        expected_histo = {bit[3](0b110): shots}
        self.assertEqual(expected_histo, pad.test(shots))

    def test_superdense(self):
        from .integ.meta_noinfer import superdense
        shots = 1024
        expected_histos = (
            {bit[2](0b00): shots},
            {bit[2](0b01): shots},
            {bit[2](0b10): shots},
            {bit[2](0b11): shots},
        )
        self.assertEqual(expected_histos, superdense.test(shots))

    def test_superdense_noidflip(self):
        from .integ.meta_noinfer import superdense_noidflip
        shots = 1024
        expected_histos = (
            {bit[2](0b00): shots},
            {bit[2](0b01): shots},
            {bit[2](0b10): shots},
            {bit[2](0b11): shots},
        )
        self.assertEqual(expected_histos, superdense_noidflip.test(shots))

    def test_teleport(self):
        from .integ.meta_noinfer import teleport
        shots = 1024
        expected_histos = (
            {bit[1](0b0): shots},
            {bit[1](0b1): shots},
            {bit[1](0b0): shots},
            {bit[1](0b1): shots},
        )
        self.assertEqual(expected_histos, teleport.test(shots))

    def test_tilt(self):
        from .integ.meta_noinfer import tilt
        shots = 1024
        expected_histos = (
            {bit[1](0b0): shots},
            {bit[1](0b1): shots},
            {bit[1](0b0): shots},
            {bit[1](0b1): shots},
        )
        self.assertEqual(expected_histos, tilt.test(shots))

    def test_ij(self):
        from .integ.meta_noinfer import ij
        shots = 1024
        expected_histos = (
            {bit[1](0b0): shots},
            {bit[1](0b1): shots},
            {bit[1](0b0): shots},
            {bit[1](0b1): shots},
        )
        self.assertEqual(expected_histos, ij.test(shots))

    def test_fourier(self):
        from .integ.meta_noinfer import fourier
        shots = 1024
        expected_histo = {bit[3](0b101): shots}
        self.assertEqual(expected_histo, fourier.test(shots))

    def test_grover(self):
        from .integ.meta_noinfer import grover
        shots = 1024
        expected_meas = bit[4](0b1010)
        actual_histo = grover.test(shots)
        self.assertGreater(actual_histo.get(expected_meas, 0),
                           shots//4*3, "Too few correct answers")

    @unittest.skip("gets stuck because we cannot infer an internal dim var")
    def test_period(self):
        from .integ.meta_noinfer import period
        shots = 1024
        self.assertTrue(any(period.test() == 4 for _ in range(32)))

@unittest.skipIf(should_skip, skip_msg)
class MetaInferIntegrationTests(unittest.TestCase):
    """
    Integration tests that use full metaQwerty features, including type
    inference.
    """

    def setUp(self):
        _reset_compiler_state()

    def test_infer_ret_type(self):
        from .integ.meta import infer_ret_type
        shots = 1024
        expected_histo = {bit[3](0b111): shots}
        self.assertEqual(expected_histo, infer_ret_type.test(shots))

    def test_infer_ret_type_tensor(self):
        from .integ.meta import infer_ret_type_tensor
        shots = 1024
        expected_histo = {bit[3](0b111): shots}
        self.assertEqual(expected_histo, infer_ret_type_tensor.test(shots))

@unittest.skipIf(should_skip, skip_msg)
class QCE25FigureIntegrationTests(unittest.TestCase):
    """The figures from the QCE '25 paper as integration tests."""

    def setUp(self):
        _reset_compiler_state()

    def test_fig1_fig2_grover(self):
        from .integ.qce25_figs import grover
        self.assertGreater(sum(grover.test() == '1010' for _ in range(32)), 20)

    @unittest.skip("ensemble operator not yet implemented")
    def test_fig3_superpos_vs_ensemble(self):
        from .integ.qce25_figs import superpos_vs_ensemble
        shots = 1024
        actual_superpos_histo, actual_ensemble_histo = \
            superpos_vs_ensemble.test(shots)

        zero, one = bit[1](0b0), bit[1](0b1)
        self.assertGreater(actual_ensemble_histo.get(zero, 0), shots//8, "Too few zeros")
        self.assertGreater(actual_ensemble_histo.get(one, 0), shots//8, "Too few ones")
        self.assertEqual(shots, actual_ensemble_histo.get(zero, 0)
                                + actual_ensemble_histo.get(one, 0),
                         "missing shots")

        expected_superpos_histo = {zero: shots}
        self.assertEqual(expected_superpos_histo, actual_superpos_histo)


    def test_fig4a_discard_invalid(self):
        from .integ.qce25_figs import discard_invalid
        with self.assertRaisesRegex(QwertyTypeError,
                                    'exactly once'):
            discard_invalid.test()

    def test_fig4b_discard_valid(self):
        from .integ.qce25_figs import discard_valid
        actual = {discard_valid.test() for _ in range(32)}
        self.assertIn(bit[1](0b0), actual)
        self.assertIn(bit[1](0b1), actual)

    def test_fig5_superdense(self):
        from .integ.qce25_figs import superdense
        for payload_int in range(1 << 2):
            payload = bit[2](payload_int)
            for i in range(4):
                expected_output = payload
                actual_output = superdense.test(payload)
                self.assertEqual(expected_output, actual_output)

    def test_fig6_fig7_prelude(self):
        from .integ.qce25_figs import prelude
        shots = 1024
        expected_histo = {bit[1](0b1): shots}
        actual_histo = prelude.test(shots)
        self.assertEqual(expected_histo, actual_histo)

    def test_fig9_grovermeta(self):
        from .integ.qce25_figs import grovermeta
        for _ in range(32):
            expected_output = bit[4](0b1010)
            actual_output = grovermeta.test()
            self.assertEqual(expected_output, actual_output)

    @unittest.skip("cannot instantiate or infer return types")
    def test_fig10_fig11_qpe(self):
        from .integ.qce25_figs import qpeuser
        for _ in range(32):
            expected_output = 'Expected: 225.0\nActual: 225.0'
            actual_output = qpeuser.test()
            self.assertEqual(expected_output, actual_output)

    def test_fig13_teleport(self):
        from .integ.qce25_figs import teleport
        shots = 1024
        expected_histos = ({bit[1](0b0): shots},
                           {bit[1](0b1): shots},
                           {bit[1](0b0): shots},
                           {bit[1](0b1): shots},
                           {bit[1](0b0): shots},
                           {bit[1](0b1): shots})
        actual_histos = teleport.test(shots)
        self.assertEqual(expected_histos, actual_histos)

    @unittest.skip("parametric kernels, type inference not implemented")
    def test_fig14_bv(self):
        from .integ.qce25_figs import bv
        for _ in range(32):
            expected_output = '1101'
            actual_output = bv.test()
            self.assertEqual(expected_output, actual_output)

    @unittest.skip("mod not implemented in classical functions")
    def test_fig16_period(self):
        from .integ.qce25_figs import period
        self.assertTrue(any(period.test() == 'Success!' for _ in range(32)))

    @unittest.skip("cannot infer return types & modmul not implemented")
    def test_fig18_shor(self):
        from .integ.qce25_figs import shor
        self.assertTrue(any(shor.test() == 4 for _ in range(32)))
