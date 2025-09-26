import unittest
from fractions import Fraction
from qwerty import *

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

    def test_bit_eq_and_eq(self):
        x = bit[8](0b1101)
        y = bit[8](0b1101)
        self.assertEqual(x == y, bit[1](0b1))

    def test_bit_eq_and_ne(self):
        x = bit[8](0b1101)
        y = bit[8](0b1110)
        self.assertEqual(x == y, bit[1](0b0))

    def test_bit_ne_and_eq(self):
        x = bit[8](0b1101)
        y = bit[8](0b1101)
        self.assertEqual(x != y, bit[1](0b0))

    def test_bit_ne_and_ne(self):
        x = bit[8](0b1101)
        y = bit[8](0b1110)
        self.assertEqual(x != y, bit[1](0b1))

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

