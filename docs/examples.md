Qwerty Examples
===============

You can find some example Qwerty programs in the `examples/` directory. You can
run any of them like any Python script, as `./dj.py` or `python dj.py`. Some
take command-line arguments; you can see a description by passing `--help` to
any example. The examples are the following:

* `bell.py`: Prepares a [Bell state][32] and prints measurement statistics
* `ghz.py`: Prepares a [GHZ][25] state and measures it. Takes number of
  qubits as command-line argument
* `ghz_flip.py`: Same as `ghz.py` except the `.flip` primitive is used instead
  of a basis translation
* `deutsch.py`: Deutsch's algorithm from Section 1.4.3 of Nielsen and Chuang
* `bv.py`: [Bernstein–Vazirani][15], effectively a quantum "Hello World".
  Takes the secret bitstring $s$ as a command-line argument
* `bv_simple.py`: A trimmed-down version of `bv.py` that fits on a
  presentation slide or poster
* `dj.py`: [Deutsch-Jozsa][17], a more general version of
  Bernstein-Vazirani (arguably). Based on Section 1.4.4 of Nielsen and
  Chuang
* `grover.py`: [Grover's Algorithm][18], aka unstructured search. The
  oracle is pretty trivial, currently looking for all 1s. Based on Section
  6.1 of Nielsen and Chuang. The (required) command line argument is the
  number of qubits.
* `period.py`: A very fun example of finding the period $r$ of a function
  $f(x) = f(x+r)$ based on Section 5.4.1 of Nielsen and Chuang.
* `simon.py`: Simon's algorithm ([1][24], [2][26])
  * `simon_post.py`: Full [classical post-processing][27] for Simon's
* `qpe.py`: [Quantum phase estimation][19], a technique for estimating the
  eigenvalues of an operator and a crucial ingredient for order finding
  (below). Based on Section 5.2 of Nielsen and Chuang.
* `order_finding.py`: [Order finding][20], which uses `qpe.py` to
  probabilistically find the multiplicative order $r$ of an integer $x$
  mod $N$. Used in [Shor's algorithm][21] (below). Based on Section 5.3.1
  of Nielsen and Chuang.
* `shors.py`: a pure-Python implementation of [Shor's algorithm][21] which
  uses `order_finding.py`. Currently set up to factor 15, because larger
  numbers cannot be simulated feasibly. Based on Section 5.3.2 of Nielsen
  and Chuang
* `fix_pt_amp.py`: Fixed-point amplitude amplification [due to Yoder, Low,
  and Chuang][28]
  * `fix_pt_phases.py`: Classical code to generate QSP/QSVT phases used
    in the algorithm
* `match.py`: [Niroula-Nam][29] string matching (uses `fix_pt_amp.py`)
* `stern_gerlach.py`: The [Stern–Gerlach experiment][31] as described [by
   Karam (Fig. 4)][30]
* `teleport.py`: Quantum teleportation as described in Section 1.3.7 of
  Nielsen and Chuang

All examples take a `--acc` flag that is useful for testing QIR-EE integration
(see `docs/qiree.md`).

There is also a helper script, `./render.sh`, which generates `.png` files for
visualizing an AST from `.dot` files (which are generated from saying
`QWERTY_DEBUG=1 python3 an_example.py` — see `docs/debugging.md` for details).

[15]: https://en.wikipedia.org/wiki/Bernstein%E2%80%93Vazirani_algorithm
[17]: https://en.wikipedia.org/wiki/Deutsch%E2%80%93Jozsa_algorithm
[18]: https://en.wikipedia.org/wiki/Grover%27s_algorithm
[19]: https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm
[20]: https://en.wikipedia.org/wiki/Shor%27s_algorithm#Quantum_order-finding_subroutine
[21]: https://en.wikipedia.org/wiki/Shor%27s_algorithm
[24]: https://en.wikipedia.org/wiki/Simon%27s_problem
[25]: https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state
[26]: https://www.cs.cmu.edu/~odonnell/quantum15/lecture06.pdf
[27]: https://quantumcomputing.stackexchange.com/a/29407/13156
[28]: https://doi.org/10.1103/PhysRevLett.113.210501
[29]: https://www.nature.com/articles/s41534-021-00369-3
[30]: https://doi.org/10.1119/10.0000258
[31]: https://en.wikipedia.org/wiki/Stern%E2%80%93Gerlach_experiment
[32]: https://en.wikipedia.org/wiki/Bell_state
