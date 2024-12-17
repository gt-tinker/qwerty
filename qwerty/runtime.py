"""
This module contains everything for the Python Qwerty runtime _except_ anything
directly related to JIT compilation or ``@qpu``/``@classical`` kernels. This
includes:

* Python types defined to make the Python interpreter happy with type
  annotations in ``@qpu`` kernels
* Operator overloads to get ``@classical`` kernels to execute classically in
  the Python interpreter
* Python implementations of histograms and continued fractions
"""

import operator
import functools
from fractions import Fraction
from abc import ABC, abstractmethod
from typing import List, Optional, Iterable

from ._qwerty_harness import Bits, Integer, Angle, Tuple, Amplitude

class dimvar:
    """
    A dimension variable, like the ``N`` in ``std[N]``. Programmers who want a
    new dimension variable ``FOO`` can instantiate this as follows::

        FOO = dimvar('FOO')
    """

    def __init__(self, name):
        self._name = name
        self._intval = None
        self._last_explicit_val = None

    # Used for executing @classical kernels in the Python interpreter
    # (Specifically, for slicing using dimension variables)
    def __index__(self):
        return self._intval

    def __int__(self):
        return self._intval

    def __call__(self, explicit_val: int):
        self._last_explicit_val = explicit_val
        return self

    # This looks like I should be imprisoned for writing it, but unfortunately,
    # it's necessary. The Python compiler likes snooping around in type
    # annotations, even at function definition time. These overrides (1) allow
    # users to write type annotations containing dimension variable expressions
    # and (2) allow classical (Python) execution of @classical kernels that use
    # constant expression arithmetic.
    #
    # Command to generate this:
    #     pbpaste | awk 'BEGIN { FS="( |\\()" } { print $0": return self._intval."$6"(int(other))" }' | pbcopy
    # (You will need to manually fix *args for pow and rpow, though)
    def __add__(self, other): return self._intval.__add__(int(other)) if self._intval is not None else self
    def __sub__(self, other): return self._intval.__sub__(int(other)) if self._intval is not None else self
    def __mul__(self, other): return self._intval.__mul__(int(other)) if self._intval is not None else self
    def __matmul__(self, other): return self._intval.__matmul__(int(other)) if self._intval is not None else self
    def __truediv__(self, other): return self._intval.__truediv__(int(other)) if self._intval is not None else self
    def __floordiv__(self, other): return self._intval.__floordiv__(int(other)) if self._intval is not None else self
    def __mod__(self, other): return self._intval.__mod__(int(other)) if self._intval is not None else self
    def __divmod__(self, other): return self._intval.__divmod__(int(other)) if self._intval is not None else self
    def __pow__(self, *args): return self._intval.__pow__(*args) if self._intval is not None else self
    def __lshift__(self, other): return self._intval.__lshift__(int(other)) if self._intval is not None else self
    def __rshift__(self, other): return self._intval.__rshift__(int(other)) if self._intval is not None else self
    def __and__(self, other): return self._intval.__and__(int(other)) if self._intval is not None else self
    def __xor__(self, other): return self._intval.__xor__(int(other)) if self._intval is not None else self
    def __or__(self, other): return self._intval.__or__(int(other)) if self._intval is not None else self
    def __radd__(self, other): return self._intval.__radd__(int(other)) if self._intval is not None else self
    def __rsub__(self, other): return self._intval.__rsub__(int(other)) if self._intval is not None else self
    def __rmul__(self, other): return self._intval.__rmul__(int(other)) if self._intval is not None else self
    def __rmatmul__(self, other): return self._intval.__rmatmul__(int(other)) if self._intval is not None else self
    def __rtruediv__(self, other): return self._intval.__rtruediv__(int(other)) if self._intval is not None else self
    def __rfloordiv__(self, other): return self._intval.__rfloordiv__(int(other)) if self._intval is not None else self
    def __rmod__(self, other): return self._intval.__rmod__(int(other)) if self._intval is not None else self
    def __rdivmod__(self, other): return self._intval.__rdivmod__(int(other)) if self._intval is not None else self
    def __rpow__(self, *args): return self._intval.__rpow__(*args) if self._intval is not None else self
    def __rlshift__(self, other): return self._intval.__rlshift__(int(other)) if self._intval is not None else self
    def __rrshift__(self, other): return self._intval.__rrshift__(int(other)) if self._intval is not None else self
    def __rand__(self, other): return self._intval.__rand__(int(other)) if self._intval is not None else self
    def __rxor__(self, other): return self._intval.__rxor__(int(other)) if self._intval is not None else self
    def __ror__(self, other): return self._intval.__ror__(int(other)) if self._intval is not None else self

# This line is pretty disturbing but a necessary evil for the moment given how
# unexpectedly picky Python can be about type annotations. Code used to
# generate this:
#     print('; '.join("{a} = dimvar('{a}')".format(a=chr(ord('A')+i)) for i in range(26)))
A = dimvar('A'); B = dimvar('B'); C = dimvar('C'); D = dimvar('D'); E = dimvar('E'); F = dimvar('F'); G = dimvar('G'); H = dimvar('H'); I = dimvar('I'); J = dimvar('J'); K = dimvar('K'); L = dimvar('L'); M = dimvar('M'); N = dimvar('N'); O = dimvar('O'); P = dimvar('P'); Q = dimvar('Q'); R = dimvar('R'); S = dimvar('S'); T = dimvar('T'); U = dimvar('U'); V = dimvar('V'); W = dimvar('W'); X = dimvar('X'); Y = dimvar('Y'); Z = dimvar('Z')

class HybridPythonQwertyType(ABC):
    """
    Abstract base class for Python objects that can be captured by a ``@qpu``
    or ``@classical`` kernel.
    """

    @abstractmethod
    def as_qwerty_obj(self):
        """
        Return an instance of a type defined in ``_qwerty_harness.cpp`` that
        wraps an instance of the ``HybridObj`` C++ class.
        """
        ...

class bit(HybridPythonQwertyType):
    """
    An array of classical bits. Written as ``bit[3]`` for an array of 3 bits,
    for example. (In a ``@qpu`` kernel type annotation, ``bit[1]`` can be
    abbreviated as ``bit`.) An instance can be instantiated with the
    following::

        secret_bits = bit[4](0b1101)

    Or alternatively, if you want to convert a string to a bit array::

        secret_bits = bit.from_str('1101')

    This array of bits can be sliced like a Python array. The most significant
    (leftmost) bit of a ``bit[N]`` has index 0, and the least significant
    (rightmost) bit has index N-1. Slicing returns another bit instance.
    Common bitwise operations are also supported.
    """

    _n_bits_hack = None

    def __class_getitem__(cls, key):
        # Support the syntax bit[4](0b0011)
        cls._n_bits_hack = key
        return cls

    def __init__(self, as_int: int, n_bits: Optional[int] = None):
        if n_bits is None:
            # This is a bit hairy: if you write bit[4] as a type annotation
            # somewhere and then write bit(x) later, then this will behave
            # unexpectedly
            if type(self)._n_bits_hack is None:
                raise ValueError('Not specifying n_bits (the second argument '
                                 'to bit()) requires writing the number of '
                                 'bits in brackets, like bit[4](0b0011)')
            n_bits = type(self)._n_bits_hack
            type(self)._n_bits_hack = None

        if not isinstance(as_int, int):
            raise TypeError('Expected int as value of qwerty.bit, not '
                            + type(as_int).__name__)

        if not isinstance(n_bits, int):
            raise TypeError('Expected int as number of bits for qwerty.bit, '
                            'not ' + type(n_bits).__name__)
        if n_bits <= 0:
            raise ValueError('Expected positive number of bits but got '
                             + str(n_bits))

        self.as_int = as_int
        self.n_bits = n_bits

    # Bit of a hack here: to allow executing @classical kernels from Python, we
    # may need to store an oversized int. But the user shouldn't know that
    def _official_int(self):
        return self.as_int & ((1 << self.n_bits)-1)

    def __hash__(self):
        return hash((self._official_int(), self.n_bits))

    def __len__(self):
        return self.n_bits

    def __eq__(self, other):
        return isinstance(other, bit) \
               and other._official_int() == self._official_int() \
               and other.n_bits == self.n_bits

    def __str__(self):
        return '{:0{n}b}'.format(self._official_int(), n=self.n_bits)

    def __repr__(self):
        return 'qwerty.bit[{}](0b{:0{}b})'.format(self.n_bits,
                                                  self._official_int(),
                                                  self.n_bits)

    def __bool__(self):
        return bool(self._official_int())

    def __int__(self):
        return self._official_int()

    def __lt__(self, other):
        return int(self) < int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def _get_lower_upper(self, idx):
        if isinstance(idx, slice):
            lower, upper, step = idx.indices(self.n_bits)
        else: # int
            idx = int(idx)
            if idx < 0 or idx >= self.n_bits:
                raise ValueError('index out of range')
            lower, upper, step = idx, idx+1, 1

        if step != 1:
            raise ValueError('Unsupported step != 1 on bit')
        # Seems like built-in Python types just return an empty result for this
        # case, but I think it's more useful to throw an error
        if lower >= self.n_bits or upper > self.n_bits:
            raise ValueError('Slice out of range')
        return lower, upper

    def __getitem__(self, idx):
        lower, upper = self._get_lower_upper(idx)
        n_bits_wanted = upper-lower
        return bit((self.as_int >> (self.n_bits-upper)) & ~(-1 << n_bits_wanted), n_bits_wanted)

    def __setitem__(self, idx, val):
        if not isinstance(val, bit):
            return NotImplemented

        lower, upper = self._get_lower_upper(idx)
        n_bits_to_update = upper-lower

        if n_bits_to_update != val.n_bits:
            raise ValueError('length of bits on right-hand side of assignment '
                             'do not match slice size')

        # Prevent severe trolling due to possible sign bits
        masked_val = val.as_int & ~(1 << n_bits_to_update)
        # Gap between LSB and range of bits to update
        lsb_offset = self.n_bits - upper
        # We want to clear the relevant range of bits. So we want a mask like:
        #
        #     n_bits_to_update
        #            _|__  _________ lsb_offset
        #           /    \/   \
        #     11111100000011111
        #           ^     ^
        #           |     |
        #       lower     upper
        #
        # The zeros above are introduced by the left operand of | below. The
        # right | operand re-introduces the rightmost group of 1s above:
        mask = -1 << (n_bits_to_update + lsb_offset) | ~(-1 << lsb_offset)
        self.as_int = (self.as_int & mask) ^ (masked_val << lsb_offset)

    def get_bits(self):
        """
        Return an iterable of all bits in this array (each as an ``int``),
        ordered from most significant to least significant.
        """
        for i in range(self.n_bits):
            yield (self.as_int >> (self.n_bits-1-i)) & 0x1

    def concat(self, other):
        """
        Concatenate two bit arrays together, returning the result.
        """
        if not isinstance(other, bit):
            raise ValueError('Can only concat bits with other bits, not {}'
                             .format(type(bit).__name__))

        return bit(self.as_int << other.n_bits | other._official_int(),
                   self.n_bits + other.n_bits)

    @classmethod
    def from_str(cls, bits: str):
        """
        Return an array of bits whose entries correspond to the characters in a
        nonempty Python ``str`` containing only ``'0'``s and ``'1'``s.
        """
        as_int = int(bits, 2)
        return cls(as_int, len(bits))

    def as_qwerty_obj(self) -> Bits:
        # Round to next multiple of 8
        # (https://stackoverflow.com/a/1766566/321301), then divide by 8
        n_bytes = ((self.n_bits + 0b111) & ~0b111) >> 3
        as_bytes = self._official_int().to_bytes(n_bytes, byteorder='big', signed=False)
        return Bits(as_bytes, self.n_bits)

    @classmethod
    def from_qwerty_obj(cls, bits: Bits):
        as_int = int.from_bytes(bits.as_bytes(), byteorder='big', signed=False)
        return cls(as_int, bits.get_n_bits())

    def as_bin_frac(self):
        """
        Interpret this array of bits as a binary fraction, i.e., as::

            0.1101
              ^^^^

        where ``0.`` is a binary point, not a decimal point. The value of that
        expression is then ``1*1/2 + 1*1/4 + 0*1/8 + 1*1/16``.

        The result of this function is a ``fractions.Fraction`` instance.
        """
        acc = Fraction(0)
        for i, b in enumerate(self.get_bits()):
            if b:
                acc += Fraction(1, 2**(i+1))
        return acc

    # Operator overloads (and methods) needed to run a @classical function
    # inside Python (instead of embedded inside a quantum circuit inside an
    # LLVM/QIR JIT land)

    def _get_other_int(self, other):
        if isinstance(other, int):
            return other
        elif isinstance(other, dimvar):
            return other._intval
        elif isinstance(other, type(self)):
            if self.n_bits != other.n_bits:
                raise TypeError('bit size mismatch: {} != {}'
                                .format(self.n_bits, other.n_bits))
            return other.as_int
        else:
            return NotImplemented

    def __and__(self, other):
        other_int = self._get_other_int(other)
        if other_int is NotImplemented:
            return NotImplemented
        return bit(self.as_int & other_int, self.n_bits)

    def __or__(self, other):
        other_int = self._get_other_int(other)
        if other_int is NotImplemented:
            return NotImplemented
        return bit(self.as_int | other_int, self.n_bits)

    def __xor__(self, other):
        other_int = self._get_other_int(other)
        if other_int is NotImplemented:
            return NotImplemented
        return bit(self.as_int ^ other_int, self.n_bits)

    def __invert__(self):
        return bit(~self.as_int & ~(-1 << self.n_bits), self.n_bits)

    def __rand__(self, other):
        # AND is commutative
        return self.__and__(other)

    def __ror__(self, other):
        # OR is commutative
        return self.__or__(other)

    def __rxor__(self, other):
        # XOR is commutative
        return self.__xor__(other)

    def __mul__(self, other):
        other_int = self._get_other_int(other)
        if other_int is NotImplemented:
            return NotImplemented
        return bit(self.as_int * other_int, self.n_bits)

    def __rmul__(self, other):
        # Multiplication is commutative
        return self.__mul__(other)

    def __mod__(self, other):
        other_int = self._get_other_int(other)
        if other_int is NotImplemented:
            return NotImplemented
        return bit(self.as_int % other_int, self.n_bits)

    def __rmod__(self, other):
        other_int = self._get_other_int(other)
        if other_int is NotImplemented:
            return NotImplemented
        return bit(other_int % self.as_int, self.n_bits)

    def __lshift__(self, other):
        other_int = self._get_other_int(other)
        if other_int is NotImplemented:
            return NotImplemented
        return bit(self._official_int() << other_int, self.n_bits)

    def __rshift__(self, other):
        other_int = self._get_other_int(other)
        if other_int is NotImplemented:
            return NotImplemented
        return bit(self._official_int() >> other_int, self.n_bits)

    def _reduction(self, reducer):
        all_bits = ((self._official_int() >> (self.n_bits-1 - i)) & 0x1
                    for i in range(self.n_bits))
        bit_int = functools.reduce(reducer, all_bits)
        return bit(bit_int, 1)

    def xor_reduce(self):
        """
        XOR all entries in the bit array together, returning a 1-bit array with
        the result.
        """
        return self._reduction(operator.xor)

    def and_reduce(self):
        """
        AND all entries in the bit array together, returning a 1-bit array with
        the result.
        """
        return self._reduction(operator.and_)

    def rotl(self, amt):
        """
        Left circular bit shift by ``amt``.
        """
        # Avoid sign extension trolling us
        x = self._official_int()
        k = int(amt) % self.n_bits
        return bit(x << k | x >> (self.n_bits-k), self.n_bits)

    def rotr(self, amt):
        """
        Right circular bit shift by ``amt``.
        """
        # Avoid excess bits on the left
        x = self._official_int()
        k = int(amt) % self.n_bits
        return bit(x >> k | ((x << (self.n_bits-k)) & ~(-1 << self.n_bits)), self.n_bits)

    def repeat(self, amt):
        """
        Concate this bit array with itself ``amt`` times, returning the result.
        """
        amt = int(amt)
        x = self._official_int()
        result = 0
        for i in range(amt):
            result = result | (x << (i * self.n_bits))
        return bit(result, self.n_bits * amt)

class cfrac:
    """
    A representation of a continued fraction (specifically a "simple continued
    fraction," or "regular continued fraction"). This is useful for the
    classical post-processing of the quantum period finding algorithm.

    For context on what a continued fraction is, please see [1], Appendix A4.4
    of [2], or an undergraduate number theory text of your choice.

    [1]: https://en.wikipedia.org/wiki/Simple_continued_fraction
    [2]: M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum
         Information, 2010.
    """

    def __init__(self, partial_denoms):
        self.partial_denoms = list(partial_denoms)

    def __str__(self):
        return str(self.partial_denoms)

    def __repr__(self):
        return 'qwerty.cfrac({})'.format(repr(self.partial_denoms))

    def __getitem__(self, key):
        """
        Slice the entries of this continued fraction like a Python ``list``.
        """
        if isinstance(key, slice):
            return type(self)(self.partial_denoms[key])
        else:
            raise ValueError('Unsupported subscripting on cfrac')

    def as_frac(self) -> Fraction:
        """
        Return this continued fraction's value as a ``fractions.Fraction``
        instance.
        """
        acc = Fraction(self.partial_denoms[-1])
        for partial_denom in reversed(self.partial_denoms[:-1]):
            acc = partial_denom + 1/acc
        return acc

    def convergents(self) -> List[Fraction]:
        """
        Return all convergents (as ``fractions.Fraction`` instances) of this
        continued fraction.
        """
        size = len(self.partial_denoms)
        return [self[:size-i].as_frac() for i in reversed(range(size))]

    @classmethod
    def from_fraction(cls, frac: Fraction):
        """
        Construct a continued fraction from a ``fractions.Fraction` instance.
        This is an implementation of the "continued fractions algorithm" as
        defined in Theorem A4.14 and Box 5.3 of of Nielsen and Chuang.
        """
        cur_frac = frac

        if cur_frac < 0:
            raise NotImplementedError('Continued fractions of negative '
                                      'numbers not supported')

        a0 = cur_frac.numerator // cur_frac.denominator
        partial_denoms = [a0]
        cur_frac = cur_frac-a0

        if cur_frac:
            while cur_frac.numerator != 1:
               cur_frac = 1/cur_frac
               denom = cur_frac.denominator
               integral, new_numerator = divmod(cur_frac.numerator, denom)
               partial_denoms.append(integral)
               cur_frac = Fraction(new_numerator, denom)
            partial_denoms.append(cur_frac.denominator)
        return cls(partial_denoms)

class _int(HybridPythonQwertyType):
    def __init__(self, int_: int):
        self.int_ = int_

    def as_qwerty_obj(self) -> Integer:
        return Integer(self.int_)

    @staticmethod
    def from_qwerty_obj(int_: Integer):
        # Weird special case here (compared to bit.from_qwery_obj()): we
        # actually do not want the user to see this _int wrapper class since
        # it's useless to them and annoying. So just return a Python int
        return int_.as_pyint()

# Similar to _int above except for tuples
class _tuple(HybridPythonQwertyType):
    def __init__(self, elts: Iterable[HybridPythonQwertyType]):
        self.elts = elts

    def as_qwerty_obj(self) -> Tuple:
        return Tuple(elt.as_qwerty_obj() for elt in self.elts)

    @staticmethod
    def from_qwerty_obj(int_: Integer):
        # TODO: Figure out if this is worth supporting
        raise NotImplementedError('Sorry, I do not know how to convert '
                                  'tuples returned by kernels back to '
                                  'Python yet')

class angle(HybridPythonQwertyType):
    """
    An angle (i.e., a ``float``) or an array of angles. This type is currently
    intended only for use in ``@qpu`` kernel type annotations.
    """

    def __class_getitem__(cls, arg):
        return cls

    def __init__(self, float_: float):
        self.float_ = float_

    def __str__(self):
        return str(self.float_)

    def __repr__(self):
        return 'angle({})'.format(self.float_)

    def as_qwerty_obj(self) -> Angle:
        return Angle(self.float_)

    @staticmethod
    def from_qwerty_obj(theta: Angle):
        # Same special case as _int.from_qwerty_obj() above
        return theta.as_pyfloat()

class ampl(HybridPythonQwertyType):
    """
    An amplitude, i.e., a complex number, or an array of amplitudes. This type
    is currently intended only for use in ``@qpu`` kernel type annotations.
    """

    def __class_getitem__(cls, arg):
        return cls

    def __init__(self, z: complex):
        self._z = z

    def __str__(self):
        return str(self._z)

    def __repr__(self):
        return 'ampl({})'.format(self._z)

    def as_qwerty_obj(self) -> Amplitude:
        return Amplitude(self._z)

    @staticmethod
    def from_qwerty_obj(amp: Amplitude):
        # Same special case as _int.from_qwerty_obj() above
        return amp.as_pycomplex()

# TODO: how does this relate to a KernelHandle?
class qfunc:
    """
    Represents a function from qubits to qubits. (To be used only in ``@qpu``
    kernel type annotations.) This type alias is useful thanks to the following
    shorthand forms:

    1. ``qfunc`` is equivalent to ``func[[qubit[1]], qubit[1]]``
    2. ``qfunc[N]`` is equivalent to ``func[[qubit[N]], qubit[N]]``
    3. ``qfunc[M,N]`` is equivalent to ``func[[qubit[M]], qubit[N]]``
    """

    def __class_getitem__(cls, key):
        return cls

class rev_qfunc:
    """
    Similar to ``qfunc`` except explicitly flagged as reversible. The
    principle of flagging some functions as explicitly reversible is necessary
    to avoid e.g., trying to run an irreversible function ``f`` backwards with
    ``~f``. Imagine ``f`` performed a measurement, for instance.

    The following shorthand forms are supported:

    1. ``rev_qfunc`` is equivalent to ``rev_func[[qubit[1]], qubit[1]]``
    2. ``rev_qfunc[N]`` is equivalent to ``rev_func[[qubit[N]], qubit[N]]``
    3. ``rev_qfunc[M,N]`` is equivalent to ``rev_func[[qubit[M]], qubit[N]]``
    """

    def __class_getitem__(cls, key):
        return cls

class cfunc:
    """
    A classical function from bits to bits, that is, a classical analog of
    ``qfunc``. (To be used only in ``@qpu`` kernel type annotations.) This type
    alias is useful thanks to the following shorthand forms:

    1. ``cfunc`` is equivalent to ``func[[bit[1]], bit[1]]``
    2. ``cfunc[N]`` is equivalent to ``func[[bit[N]], bit[N]]``
    3. ``cfunc[M,N]`` is equivalent to ``func[[bit[M]], bit[N]]``
    """

    def __class_getitem__(cls, key):
        return cls

class func:
    """
    A type represnting a Qwerty kernel. This is to be used only in ``@qpu``
    kernel type annotations. The expected syntax for ``func`` is the
    following::

        func[[arg1_type, arg2_type, ..., argn_type], ret_type]
    """

    def __class_getitem__(cls, key):
        return cls

class rev_func:
    """
    Similar to ``func`` except explicitly flagged as reversible. (Also intended
    only to be used in ``@qpu`` kernel type annotations.) The expected syntax
    for ``rev_func`` is the following::

        rev_func[[arg1_type, arg2_type, ..., argn_type], ret_type]

    It is more likely that you would want to use ``rev_qfunc`` than this type
    directly.
    """

    def __class_getitem__(cls, key):
        return cls

class qubit:
    """
    An array of qubits, i.e., ``qubit[N]``. This is intended only to be used in
    ``@qpu`` kernel type annotations, since the Qwerty programming model does
    not allow directly manipulating qubits in classical Python code. Instead,
    qubits should be manipulated in a ``@qpu`` kernel. Writing ``qubit`` is
    accepted shorthand for ``qubit[1]``.
    """
    def __class_getitem__(cls, key):
        return cls

def reversible(f):
    """
    Type annotation used to flag ``@qpu`` kernel as being reversible. Used like
    this::

        @qpu
        @reversible
        def f(q: qubit[N]) -> qubit[N]:
            # ...
    """
    return f

def print_histogram(histogram):
    """
    Print a mapping of ``bit[N]``s to ``int`` counts as percentages to stdout.
    """
    total = sum(histogram.values())
    for meas, count in sorted(histogram.items()):
        print('{} -> {:.02f}%'.format(str(meas), count/total*100))
