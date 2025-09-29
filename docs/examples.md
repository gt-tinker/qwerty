Qwerty Examples
===============

You can find some example Qwerty programs in the `examples/` directory. You can
run any of them like any Python script, as `./dj.py` or `python dj.py`. Most
take command-line arguments; you can see a description by passing `--help` to
any example. The examples are the following:

* [`coin_flip.py`](../examples/coin_flip.py): Prepare a uniform superposition
  of 0 and 1 and measure, like flipping a coin
* [`bell.py`](../examples/bell.py): Prepares a [Bell state][1] and prints
  measurement statistics
* [`ghz.py`](../examples/ghz.py): Prepares a [GHZ][2] state (a Bell state
  generalized to more than two qubits) and measures it. Takes number of qubits
  as command-line argument
* [`deutsch.py`](../examples/deutsch.py): Deutsch's algorithm from Section
  1.4.3 of Nielsen and Chuang
* [`dj.py`](../examples/dj.py): [Deutsch-Jozsa][3], Deutsch's algorithm
  generalized to functions who take more than one bit of input. Based on
  Section 1.4.4 of Nielsen and Chuang.
* [`bv.py`](../examples/bv.py): [Bernstein–Vazirani][4], effectively a quantum
  "Hello World". Takes the secret bitstring $s$ as a command-line argument
* [`grover.py`](../examples/grover.py): [Grover's Algorithm][5], aka
  unstructured search. The oracle is pretty trivial, currently looking for all
  1s. Based on Section 6.1 of Nielsen and Chuang. The (required) command line
  argument is the number of qubits.
* [`period.py`](../examples/period.py): A fun example of finding the period of
  a function based on Section 5.4.1 of Nielsen and Chuang.
* [`simon.py`](../examples/simon.py): Simon's algorithm ([1][6], [2][7]), the
  first quantum algorithm to promise exponential speedup
  * [`simon_postprocess.py`](../examples/simon_postprocess.py): Full [classical
    post-processing][8] for Simon's
* [`teleport.py`](../examples/teleport.py): Quantum teleportation, a procedure
  for transmitting a qubit by consuming an entangled pair of qubits, as
  described in Section 1.3.7 of Nielsen and Chuang
* [`superdense.py`](../examples/superdense.py): Superdense coding, a technique
  for transmitting two classical bits using one qubit, as described in Section
  2.3 of Nielsen and Chuang
* [`qpe.py`](../examples/qpe.py): [Quantum phase estimation][9], a technique
  for estimating how much a quantum function tilts a state. This a crucial
  ingredient for some formulations of Shor's factoring algorihtm. Based on
  Section 5.2 of Nielsen and Chuang.
* [`shor.py`](../examples/shor.py): an implementation of [Shor's algorithm][10]
  with the classical portion written in Python and the quantum part written in
  Qwerty. Currently set up to factor 15, because larger numbers cannot be
  simulated feasibly. Based on Sections 5.3.1-5.3.2 of Nielsen and Chuang
* [`abbrev/`](../examples/abbrev/): These are abbreviated examples originally
  from the QCE '25 paper but intended to be shown the Qwerty website. Unlike
  examples in the parent directory, these do not contain `argparse` command
  line argument handling, for example.

All of these examples are integration tests in [the class
`ExampleIntegrationTests`](../qwerty_pyrt/python/qwerty/tests/integration_tests.py).

[1]: https://en.wikipedia.org/wiki/Bell_state
[2]: https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state
[3]: https://en.wikipedia.org/wiki/Deutsch%E2%80%93Jozsa_algorithm
[4]: https://en.wikipedia.org/wiki/Bernstein%E2%80%93Vazirani_algorithm
[5]: https://en.wikipedia.org/wiki/Grover%27s_algorithm
[6]: https://en.wikipedia.org/wiki/Simon%27s_problem
[7]: https://www.cs.cmu.edu/~odonnell/quantum15/lecture06.pdf
[8]: https://quantumcomputing.stackexchange.com/a/29407/13156
[9]: https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm
[10]: https://en.wikipedia.org/wiki/Shor%27s_algorithm
