from qwerty import *
from grover import grover

@classical
def oracle(x: bit[4]) -> bit:
  return x[0] & ~x[1] & x[2] & ~x[3]

print(grover(oracle, num_iter=3))
