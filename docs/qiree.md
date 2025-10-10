Building with QIR-EE Support
============================

[QIR-EE][1] is a useful tool for targeting classical simulators such as
[QSim][3] or NISQ hardware via [XACC][2]. Beware: the integration between
QIR-EE and Qwerty is currently experimental.

There are three possible configurations of Qwerty and QIR-EE:

Config 0: Qwerty without QIR-EE
-------------------------------

The default configuration of the Qwerty compiler is not to include QIR-EE at
all. This way, the only built-in execution is local classical simulation via
[qir-runner][4]. The build process for this configuration is as described in
[the top-level README](../README.md).

Config 1: Qwerty with QIR-EE, but without XACC
----------------------------------------------

The simplest QIR-EE configuration is using [Qsim][3], a classical simulator.

### Building the Qwerty Compiler with QIR-EE Support

You only need to enable the `qiree` feature for the `qwerty_pyrt` crate. This
means following the build steps in [the top-level README](../README.md) except
using the following `maturin` command instead:

    $ maturin develop -F qiree -vvv

### Running a Qwerty Program with QIR-EE

When invoking a Qwerty `@qpu` function, say `kernel`, pass an accelerator
supported by QIR-EE as the `acc` keyword argument as in `kernel(acc='qsim')`.
The example Qwerty programs in [`examples/`](../examples/) also allow specifing
an accelerator on the command line as follows:

    $ cd examples
    $ python bell.py --acc qsim
    00 -> 50.00%
    11 -> 50.00%

Classical control flow is currently not supported, so the [quantum
teleportation](../examples/teleport.py) and [superdense
coding](../examples/superdense.py) examples will not work with QIR-EE yet.

Config 2: Qwerty with QIR-EE, but with XACC
-------------------------------------------

I have not tackled this yet because the Python wheel built for QIR-EE+Qwerty
would not be portable. This is because of the following:

1. It is dependent on many XACC shared libraries in `~/.xacc`
2. As such, the `rpath` is hardcoded to look up libraries in that path.
   Otherwise, you would need to obnoxiously set `$LD_LIBRARY_PATH` to
   `~/.xacc/lib` every time you ran a Qwerty example

To address this, we would need to either allow XACC to be built as a static
library (extremely difficult) or packaging up these shared objects in the
wheel.

[1]: https://arxiv.org/abs/2404.14299
[2]: https://arxiv.org/abs/1911.02452
[3]: https://github.com/quantumlib/qsim
[4]: https://github.com/qir-alliance/qir-runner/
