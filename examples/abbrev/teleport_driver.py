from qwerty import *
from teleport import teleport

@qpu
def teleport_1():
    alice = '1'
    bob = teleport(alice)
    return std.measure(bob)

histogram(teleport_1(shots=1024))

@qpu
def teleport_j():
    alice = 'j'
    bob = teleport(alice)
    return ij.measure(bob)

histogram(teleport_j(shots=1024))
