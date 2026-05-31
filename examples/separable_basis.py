#!/usr/bin/env python3
"""
Z (x) X separable basis, with per-vector phases at various angles.

{'0p','0m','1p','1m'} is separable: qubit 0 in Z, qubit 1 in X.

"""

from qwerty import *


@qpu
def no_phase() -> bit[2]:
    return '00' | {'0p', '0m', '1p', '1m'}.measure

@qpu
def all_phase() -> bit[2]:
    return '00' | {'0p'@90, '0m'@90, '1p'@90, '1m'@90}.measure


@qpu
def with_phase_180() -> bit[2]:
    return '00' | {'0p', '0m', '1p', '1m'@180}.measure


@qpu
def with_phase_45() -> bit[2]:
    return '00' | {'0p', '0m', '1p', '1m'@45}.measure


@qpu
def with_phase_60() -> bit[2]:
    return '00' | {'0p', '0m', '1p', '1m'@60}.measure


if __name__ == '__main__':
    # All four must agree (angle-independent under measurement).
    print("no phase:       ", histogram(no_phase(shots=512)))
    print("all phase:       ", histogram(all_phase(shots=512)))
    print("with phase @180:", histogram(with_phase_180(shots=512)))
    print("with phase @45: ", histogram(with_phase_45(shots=512)))
    print("with phase @60: ", histogram(with_phase_60(shots=512)))
