import os
import sys
import tempfile
import unittest
import qwerty.kernel
from qwerty.runtime import bit
from qwerty.err import QwertyTypeError

should_skip = bool(os.environ.get('SKIP_INTEGRATION_TESTS'))
skip_msg = "Skipping integration tests as requested by $SKIP_INTEGRATION_TESTS"

@unittest.skipIf(should_skip, skip_msg)
class NoMetaIntegrationTests(unittest.TestCase):
    """Integration tests that do not use metaQwerty features at all."""

    def setUp(self):
        qwerty.kernel._reset_compiler_state()

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

    def test_classical_call(self):
        from .integ.nometa import classical_call
        secret_string = bit[3](0b110)
        expected_output = [bit[1](0b0), # <- 0b000
                           bit[1](0b0), # <- 0b001
                           bit[1](0b1), # <- 0b010
                           bit[1](0b1), # <- 0b011
                           bit[1](0b1), # <- 0b100
                           bit[1](0b1), # <- 0b101
                           bit[1](0b0), # <- 0b110
                           bit[1](0b0)] # <- 0b111
        self.assertEqual(expected_output, classical_call.test(secret_string))

@unittest.skipIf(should_skip, skip_msg)
class MetaNoPreludeNoInferIntegrationTests(unittest.TestCase):
    """
    Integration tests that do use metaQwerty features but have the default
    prelude disabled.
    """

    def setUp(self):
        qwerty.kernel._reset_compiler_state()

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
        qwerty.kernel._reset_compiler_state()

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

    def test_period(self):
        from .integ.meta_noinfer import period
        expected_period = 4
        self.assertTrue(any(period.test() == expected_period for _ in range(8)))

    def test_arith_select(self):
        from .integ.meta_noinfer import arith_select
        output = arith_select.test()
        self.assertIn(output, [bit[1](0b0), bit[1](0b1)], "Final output was not in the select options!")

@unittest.skipIf(should_skip, skip_msg)
class MetaInferIntegrationTests(unittest.TestCase):
    """
    Integration tests that use full metaQwerty features, including type
    inference.

    For more information on each test, see the docstring on the module
    containing Qwerty code imported in each test.
    """

    def setUp(self):
        qwerty.kernel._reset_compiler_state()

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

    def test_period(self):
        from .integ.meta import period
        expected_period = 4
        self.assertTrue(any(period.test() == expected_period for _ in range(8)))

    def test_predicate_in(self):
        from .integ.meta import predicate_in
        shots = 1024
        expected_histo = {bit[3](0b101): shots}
        actual_histo = predicate_in.test(shots)
        self.assertEqual(expected_histo, actual_histo)

    def test_float_expr(self):
        from .integ.meta import float_expr
        shots = 1024
        expected_histos = ({bit[1](0b1): shots},
                           {bit[1](0b1): shots},
                           {bit[1](0b1): shots})
        actual_histos = float_expr.test(shots)
        self.assertEqual(expected_histos, actual_histos)

    def test_megaperm(self):
        from .integ.meta import megaperm
        shots = 1024

        # The 10-qubit case is handled by tweedledum-based synthesis, but the
        # 11-qubit case is handled by our custom synthesis.
        for n_qubits in (10, 11):
            all_ones = bit.from_str('1'*n_qubits)
            all_ones_except_upper3 = bit.from_str('000' + '1'*(n_qubits-3))
            expected_histos = ({all_ones: shots},
                               {all_ones_except_upper3: shots})
            actual_histos = megaperm.test(n_qubits, shots)
            self.assertEqual(expected_histos, actual_histos)

    def test_qft(self):
        from .integ.meta import qft
        shots = 1024
        expected_histos = ({bit[3](0b000): shots},
                           {bit[3](0b000): shots})
        actual_histos = qft.test(shots)
        self.assertEqual(expected_histos, actual_histos)

    def test_repeat(self):
        from .integ.meta import repeat
        shots = 1024
        expected_result = (bit[4](0b1111), {bit[5](0b11111): shots})
        actual_result = repeat.test(shots)
        self.assertEqual(expected_result, actual_result)

    def test_concat(self):
        from .integ.meta import concat
        shots = 1024
        expected_result = (bit[5](0b010_01), {bit[10](0b010_01_010_01): shots})
        actual_result = concat.test(shots)
        self.assertEqual(expected_result, actual_result)

    def test_slice(self):
        from .integ.meta import slice
        shots = 1024
        x = bit[5](0b01_110)
        expected_result = (x, {x.repeat(2): shots})
        actual_result = slice.test(shots)
        self.assertEqual(expected_result, actual_result)

@unittest.skipIf(should_skip, skip_msg)
class ExampleIntegrationTests(unittest.TestCase):
    """
    The example programs from the ``examples/`` directory as integration
    tests. Useful for avoiding embarassment caused by our own examples being
    broken.
    """

    def setUp(self):
        qwerty.kernel._reset_compiler_state()

    def test_coin_flip(self):
        from .integ.examples import coin_flip
        shots = 1024
        actual_histo = coin_flip.test(shots)
        zero, one = bit[1](0b0), bit[1](0b1)
        self.assertGreater(actual_histo.get(zero, 0), shots//8, "Too few zeros")
        self.assertGreater(actual_histo.get(one, 0), shots//8, "Too few ones")
        self.assertEqual(shots, actual_histo.get(zero, 0) + actual_histo.get(one, 0), "missing shots")

    def test_bell(self):
        from .integ.examples import bell
        shots = 1024
        actual_histo = bell.test(shots)
        zero, one = bit[2](0b00), bit[2](0b11)
        self.assertGreater(actual_histo.get(zero, 0), shots//8, "Too few zeros")
        self.assertGreater(actual_histo.get(one, 0), shots//8, "Too few ones")
        self.assertEqual(shots, actual_histo.get(zero, 0) + actual_histo.get(one, 0), "missing shots")

    def test_ghz(self):
        from .integ.examples import ghz
        shots = 1024
        actual_histo = ghz.test(7, shots)
        zero, one = bit[7](0b0000000), bit[7](0b1111111)
        self.assertGreater(actual_histo.get(zero, 0), shots//8, "Too few zeros")
        self.assertGreater(actual_histo.get(one, 0), shots//8, "Too few ones")
        self.assertEqual(shots, actual_histo.get(zero, 0) + actual_histo.get(one, 0), "missing shots")

    def test_deutsch(self):
        from .integ.examples import deutsch
        shots = 1024
        balanced_out = bit[1](0b1)
        constant_out = bit[1](0b0)
        expected_result = [(balanced_out, {balanced_out: shots}),
                           (constant_out, {constant_out: shots})]
        self.assertEqual(expected_result, deutsch.test(shots))

    def test_dj(self):
        from .integ.examples import dj
        shots = 1024
        # Proof by Quirk that 0b1001 is expected for balanced:
        # https://algassert.com/quirk#circuit={%22cols%22:[[%22H%22,%22H%22,%22H%22,%22H%22,%22H%22],[%22%E2%80%A2%22,1,1,1,%22X%22],[1,1,1,%22%E2%80%A2%22,%22X%22],[1,1,1,1,%22X%22],[%22H%22,%22H%22,%22H%22,%22H%22]],%22init%22:[0,0,0,0,1]}
        expected_result = [('constant', {bit[4](0b0000): shots}),
                           ('balanced', {bit[4](0b1001): shots})]
        self.assertEqual(expected_result, dj.test(shots))

    # Tests that the Qwerty AST printed when $QWERTY_DEBUG==1 runs as a Python
    # module
    def test_bv_qwerty_debug_ast(self):
        from .integ.examples import bv
        shots = 1024
        expected_histo = {bit[4](0b1101): shots}

        old_QWERTY_DEBUG = qwerty.kernel.QWERTY_DEBUG
        old_cwd = os.getcwd()
        old_sys_path = sys.path
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                qwerty.kernel.QWERTY_DEBUG = True
                os.chdir(tmpdir)
                actual_histo_normal = bv.test(shots)

                sys.path.append(os.path.join(tmpdir, 'qwerty-debug'))
                import qwerty_ast
            finally:
                os.chdir(old_cwd)
                qwerty.kernel.QWERTY_DEBUG = old_QWERTY_DEBUG
                sys.path = old_sys_path

            actual_histo_gen_ast = qwerty_ast._run_ast(shots=shots)

        self.assertEqual(expected_histo, actual_histo_normal)
        self.assertEqual(expected_histo, actual_histo_gen_ast)

    def test_grover(self):
        from .integ.examples import grover
        num_qubits = 5
        shots = 1024
        answer = bit[5](0b11111)
        expected_histo = {answer: shots}
        expected_answers = [answer]
        actual_histo, actual_answers = grover.test(num_qubits, shots)
        self.assertEqual(expected_answers, actual_answers)
        self.assertGreater(actual_histo.get(answer, 0), 9*shots//10,
                           "Too few correct answers")

    def test_period(self):
        from .integ.examples import period
        num_qubits = 7
        mod = 16
        attempts = 16
        self.assertIn(mod, set(period.test(num_qubits, mod, attempts)),
                      "Did not find correct period")

    def test_simon(self):
        from .integ.examples import simon
        num_qubits = 8
        secret_str = bit[8](1 << (num_qubits-1))
        expected_classical, expected_quantum = secret_str, secret_str
        actual_classical, actual_quantum = simon.test(num_qubits,
                                                      num_attempts=32)
        self.assertEqual(expected_classical, actual_classical)
        self.assertEqual(expected_quantum, actual_quantum)

    def test_teleport(self):
        from .integ.examples import teleport
        shots = 1024
        expected_histos = ({bit[1](0b1): shots},
                           {bit[1](0b0): shots})
        actual_histos = teleport.test(shots)
        self.assertEqual(expected_histos, actual_histos)

    def test_superdense(self):
        from .integ.examples import superdense
        shots = 1024
        expected_histos = [{bit[2](0b00): shots},
                           {bit[2](0b01): shots},
                           {bit[2](0b10): shots},
                           {bit[2](0b11): shots}]
        actual_histos = list(superdense.test(shots))
        self.assertEqual(expected_histos, actual_histos)

    def test_qpe(self):
        from .integ.examples import qpe
        shots = 1024
        angle_deg = 292.5
        prec = 4
        actual_histo = qpe.test(angle_deg, prec, shots)
        self.assertGreater(actual_histo.get(angle_deg, 0), shots*15//16,
                           "Too few correct answers")
        self.assertEqual(shots, sum(actual_histo.values()), "missing shots")

    def test_shor(self):
        from .integ.examples import shor
        number = 15
        actual_factor = shor.test(number, num_attempts=32)
        self.assertIn(actual_factor, [3, 5],
                      "Did not find correct factor")

@unittest.skipIf(should_skip, skip_msg)
class QCE25FigureIntegrationTests(unittest.TestCase):
    """The figures from the QCE '25 paper as integration tests."""

    def setUp(self):
        qwerty.kernel._reset_compiler_state()

    def test_fig1_fig2_grover(self):
        from .integ.qce25_figs import grover
        # Test that it runs
        result = grover.test_runs()
        self.assertEqual(len(result), 4)
        self.assertTrue(set(result) == {'0','1'})

        # Test that it's correct
        shots = 1024
        expected_meas = bit[4](0b1010)
        actual_histo = grover.test_correct(shots)
        self.assertGreater(actual_histo.get(expected_meas, 0),
                           shots//4*3, "Too few correct answers")

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

    def test_fig3_superpos_vs_ensemble_prob(self):
        from .integ.qce25_figs import superpos_vs_ensemble_prob
        shots = 1024
        actual_superpos_histo, actual_ensemble_histo = \
            superpos_vs_ensemble_prob.test(shots)

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
        shots = 1024
        actual_histo = discard_valid.test(1024)

        zero, one = bit[1](0b0), bit[1](0b1)
        self.assertGreater(actual_histo.get(zero, 0), shots//4, "Too few zeros")
        self.assertGreater(actual_histo.get(one, 0), shots//4, "Too few ones")
        self.assertEqual(shots, actual_histo.get(zero, 0)
                                + actual_histo.get(one, 0),
                         "missing shots")

    def test_fig5_superdense_runs(self):
        from .integ.qce25_figs import superdense
        actual_meas = superdense.test(bit[2](0b10))
        self.assertTrue(isinstance(actual_meas, bit))

    def test_fig5_superdense_correct(self):
        from .integ.qce25_figs import superdense_shots
        shots = 1024

        for payload_int in range(1 << 2):
            payload = bit[2](payload_int)
            actual_histo = superdense_shots.test(payload, shots)
            expected_histo = {payload: shots}
            self.assertEqual(expected_histo, actual_histo)

    def test_fig6_fig7_prelude(self):
        from .integ.qce25_figs import prelude
        shots = 1024
        expected_histo = {bit[1](0b1): shots}
        actual_histo = prelude.test(shots)
        self.assertEqual(expected_histo, actual_histo)

    def test_fig9_grovermeta_runs(self):
        from .integ.qce25_figs import grovermeta
        actual_output = grovermeta.test()
        self.assertTrue(isinstance(actual_output, bit))

    def test_fig9_grovermeta_correct(self):
        from .integ.qce25_figs import grovermeta_shots
        shots = 1024
        expected_meas = bit[4](0b1010)
        actual_histo = grovermeta_shots.test(shots)
        self.assertGreater(actual_histo.get(expected_meas, 0),
                           shots//4*3, "Too few correct answers")

    def test_fig10_fig11_qpe(self):
        from .integ.qce25_figs import qpeuser
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

    def test_fig14_bv_runs(self):
        from .integ.qce25_figs import bv
        expected_output = '1101'
        actual_output = bv.test()
        self.assertEqual(expected_output, actual_output)

    def test_fig14_bv_correct(self):
        from .integ.qce25_figs import bv_shots
        shots = 1024
        expected_histo = {bit[4](0b1101): shots}
        actual_histo = bv_shots.test(shots)
        self.assertEqual(expected_histo, actual_histo)

    def test_fig16_period(self):
        from .integ.qce25_figs import period
        self.assertTrue(any(period.test() == 'Success!' for _ in range(8)))

    def test_fig18_shor_runs(self):
        from .integ.qce25_figs import shor
        shor.test()
        # TODO: test that this produces the right answer once we have optimized
        #       the compiler/runtime better
