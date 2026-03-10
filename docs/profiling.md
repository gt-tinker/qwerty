Profiling the Qwerty Compiler
=============================

Profiling the Qwerty compiler can be an involved task. Qwerty programs are
started in Python, which calls Rust code, ultimately descending through C++ and
MLIR. There are no doubt other ways of approaching profiling, however, the
approach described below we found to be the most versatile, providing good
details throughout the language layers.


Approach
--------

This approach uses `py-spy`, a python profiling program that drills well into
C++ and Rust stacks. `py-spy` works by observing -- but not injecting itself
into -- the program that it profiles. As most test programs finish quickly, and
py-spy seems to have problems sampling at more than 25-50 Hz, it is necessary
to run a sample program over many iterations. The result is not a profile of a
single run, but the average of many. These results are great for providing an
overview. That being said, if the goal is to obtain a high accuracy profile of
particular segment occupying a small portion of total runtime, py-spy may not
be the ideal choice.

Guide
-----

### Install `py-spy`

Install `py-spy` using pip, making sure you are inside the Qwerty virtual
environment as described in the [README](../README.md):

    $ pip install py-spy

### Create Test Script

Create a test Python script:

1. Place the program you would like to sample in a for loop.
2. The number of iterations of the for loop depends on the desired granuality.
   For a rough overview, 50 iterations is sufficient. 500 or 5000 iterations is
   more suitable for detailed reports. Longer running programs may need lower
   numbers of iterations.
3. Place `_reset_compiler_state()` at the top of the loop body.

As an example, here is what a test script for
[`examples/bv.py`](../examples/bv.py) might look like. Note that only the code
in the body of `if __name__ == 'main':` has been changed:

**`bv-sampler.py`**:

    #!/usr/bin/env python3

    """
    Perform the Bernstein–Vazirani algorithm on a provided bitstring, printing the
    bitstring determined by both the quantum algorithm and classical algorithm.
    """

    from argparse import ArgumentParser
    from qwerty import *
    from qwerty.kernel import _reset_compiler_state

    def bv(oracle, acc=None):
        @qpu[[N]]
        def kernel():
            return ('p'**N | oracle.sign
                           | pm**N >> std**N
                           | measure**N)

        return kernel(acc=acc)

    def get_black_box(secret_string):
        @classical[[N]]
        def f(x: bit[N]) -> bit:
            return (secret_string & x).xor_reduce()
        return f

    def naive_classical(f, n_bits):
        secret_found = bit[n_bits](0)
        x = bit[n_bits](0b1 << (n_bits-1))
        for i in range(n_bits):
            secret_found[i] = f(x)
            x = x >> 1
        return secret_found

    # DIFFERENCE
    if __name__ == '__main__':
        # Run 5000 samples of bv
        for i in range(0, 5000):
            # IMPORTANT
            _reset_compiler_state()
            secret_str = bit.from_str("1000")
            n_bits = len(secret_str)
            black_box = get_black_box(secret_str)

            print('Classical:', naive_classical(black_box, n_bits))
            print('Quantum:  ', bv(black_box, acc=None))

`_reset_compiler_state()` is needed due to an oversight with the current
compiler. If this line is not placed, the compiler will crawl to a halt under
the weight of unfreed internal resources. This is an issue that we plan to
address in the future.

### Run Script

To begin sampling, run:

    $ py-spy record -o py-spy-results.prof --native --format speedscope -- python my-sampler.py

The flag `--native` is essential so that `py-spy` accurately profiles C++ and
Rust portions of the qwerty compiler.

## Visualizing with Speedscope

Visit [speedscope.app](https://www.speedscope.app/) and upload the `.prof` file
created in the previous step.

Once the profile is uploaded, select "left heavy" in the top-left corner. The
default organization will show every run one after the other. Selecting "leave
heavy" consolidates function calls together. It is very important to select
this mode as we are using a statistical approach.

If you face issues with Speedscope, try running the program in Google Chrome.
