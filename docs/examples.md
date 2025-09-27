Qwerty Examples
===============

You can find some example Qwerty programs in the `examples/` directory. You can
run any of them like any Python script, as `./dj.py` or `python dj.py`. Most
take command-line arguments; you can see a description by passing `--help` to
any example. The examples are the following:

* `coin_flip.py`: Prepare a uniform superposition of 0 and 1 and measure, like
  flipping a coin
* `bell.py`: Prepares a [Bell state][1] and prints measurement statistics
* `ghz.py`: Prepares a [GHZ][2] state (a Bell state generalized to more than
  two qubits) and measures it. Takes number of qubits as command-line argument
* `deutsch.py`: Deutsch's algorithm from Section 1.4.3 of Nielsen and Chuang
* `dj.py`: [Deutsch-Jozsa][3], Deutsch's algorithm generalized to functions
  who take more than one bit of input . Based on Section 1.4.4 of Nielsen and
  Chuang
* `bv.py`: [Bernstein–Vazirani][4], effectively a quantum "Hello World".
  Takes the secret bitstring $s$ as a command-line argument
* `grover.py`: [Grover's Algorithm][5], aka unstructured search. The
  oracle is pretty trivial, currently looking for all 1s. Based on Section
  6.1 of Nielsen and Chuang. The (required) command line argument is the
  number of qubits.

[1]: https://en.wikipedia.org/wiki/Bell_state
[2]: https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state
[3]: https://en.wikipedia.org/wiki/Deutsch%E2%80%93Jozsa_algorithm
[4]: https://en.wikipedia.org/wiki/Bernstein%E2%80%93Vazirani_algorithm
[5]: https://en.wikipedia.org/wiki/Grover%27s_algorithm
