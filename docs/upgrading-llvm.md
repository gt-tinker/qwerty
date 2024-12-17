Upgrading LLVM
==============

Updating LLVM is not the most fun thing in the world, but on the bright side,
it often brings bugfixes and improved APIs. The first step of updating the LLVM
dependency is to update the LLVM submodule we use to track the current LLVM commit
we're compatible with (CIRCT does something similar):

    $ git update-index --cacheinfo 160000,e21dc4bd5474d04b8e62d7331362edcc5648d7e5,tpls/llvm

The next step is replacing every occurrence of `llvm19`, `19_1_6`, and
`19.1.6` in `README.md` and all `docs/*.md` files. This is useful (make
sure to check the results are right):

    $ git help sed
    'sed' is aliased to '! git ls-files -s -z | grep -z -v '^16' | cut -z -f 2- | xargs -0 sed -i -e'
    $ git sed 's/\<llvm18\>/llvm19/g' -e 's/v18_1_8/v19_1_6/g' -e 's/\<18_1_8\>/19_1_6/g' -e 's/\<18\.1\.8/19.1.6/g'

To rebuild the tarballs, see the scripts under `docs/scripts/`. `build-llvm.sh`
has been tested on macOS and GNU/Linux, and
`build-llvm-for-windows/build-llvm.ps1` is for Windows. Note it's a solid idea
to delete `_skbuild` if you are testing a new LLVM release, since the existing
version may reference the previous LLVM version. (Also, for Windows,
`docs/scripts/build-llvm-for-windows/setup-env.bat` sets a lot of annoying
variables useful for building Qwerty on Windows.)

Finally, for my own convenience when copy-pasting:

    $ aws s3 cp llvm_mlir_rel_v19_1_6_x86_linux.tar.xz s3://austinjadams-com-test/qwerty/
