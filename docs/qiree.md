Building with QIR-EE/XACC Support
=================================

[QIR-EE][1] is a useful tool for targeting NISQ hardware with QIR via
[XACC][2]. Qwerty can pass QIR to QIR-EE, but it takes some work to make them
work together. Note the integration between QIR-EE and Qwerty is currently
experimental.

Step 1: Build XACC
------------------

1. Clone [the ORNL fork of the XACC git repository][3]
2. Do the classic inside the cloned XACC repo:
   ```
   $ mkdir build && cd build
   $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ..
   $ ninja install
   ```

Step 2: Build QIR-EE
--------------------

1. Check out the `qwerty` branch of [my fork][7] of the [QIR-EE git
   repository][4]
2. Set up your LLVM environment variables properly as described in the Qwerty
   build instructions
3. Run the following insided the cloned repo:
   ```
   $ mkdir build && cd build
   $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DQIREE_BUILD_TESTS=FALSE -DXACC_DIR=$HOME/.xacc -DBUILD_SHARED_LIBS=FALSE -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ..
   $ ninja install
   ```
   The `-DBUILD_SHARED_LIBS=FALSE -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE`
   portion is pretty crucial to avoid many linker errors related to both
   QIR-EE and Qwerty using LLVM. Turning off the tests stops CMake from going
   on a fishing expedition to find googletest. The path passed as `XACC_DIR`,
   `~/.xacc`, is the default XACC install dir. Heads up, if you encounter weird
   build problems, I would recommend deleting and re-creating the `build`
   directory _and_ doing `rm -rvf ~/.xacc/include/qiree*` (somehow old
   installed QIREE headers take precedence over ones in your working tree).

   If you want to build the tests, try this command line instead:
   ```
   $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGTest_ROOT=$HOME/bin/gtest -DXACC_DIR=$HOME/.xacc -DBUILD_SHARED_LIBS=FALSE -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ..
   ```
   which assumes you previously built googletest like this:
   ```
   $ git clone https://github.com/google/googletest.git
   $ mkdir googletest/build && cd googletest/build
   $ cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$HOME/bin/gtest -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ..
   $ ninja install
   ```
   The `-fPIC` flag is really important to avoid linker errors.

Step 3: Build Qwerty
--------------------

1. With the virtual environment activated and your LLVM environment variables
   set up as in step (3) above, try:
   ```
   $ QWERTY_USE_QIREE=true QIREE_DIR=$HOME/.xacc/ XACC_DIR=$HOME/.xacc/ pip install -v .
   ```
   I set `$QIREE_DIR` to `~/.xacc` is because that's where QIR-EE seems to
   install itself by default.
2. Run a simple example and check that QIR-EE actually ran:
   ```
   $ QWERTY_DEBUG=1 python examples/ghz_flip.py 3 qpp
   $ cat qiree.log
   tuple ret length 3 distinct results 2
   tuple ret result 000 count 1061
   tuple ret result 111 count 987
   tuple discarded length 0 distinct results 0
   ```

Note that right now the Python wheel built for QIR-EE+Qwerty is **_not_**
portable. This is because of the following:

1. It is dependent on many XACC shared libraries in `~/.xacc`
2. As such, the `rpath` is hardcoded to look up libraries in that path.
   Otherwise, you would need to obnoxiously set `$LD_LIBRARY_PATH` to
   `~/.xacc/lib` every time you ran a Qwerty example

We need to address this by either allowing XACC to be built as a static library
(extremely difficult) or packaging up these shared objects in the wheel.

[1]: https://arxiv.org/abs/2404.14299
[2]: https://arxiv.org/abs/1911.02452
[3]: https://github.com/ORNL-QCI/xacc/
[4]: https://github.com/ORNL-QCI/qiree
[7]: https://github.com/ausbin/qiree/tree/qwerty
