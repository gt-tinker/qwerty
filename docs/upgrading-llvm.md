Upgrading LLVM
==============

Updating LLVM is not the most fun thing in the world, but on the bright side,
it often brings bugfixes and improved APIs. The first step of updating the LLVM
dependency is to update the LLVM submodule we use to track the current LLVM commit
we're compatible with (CIRCT does something similar):

    $ git update-index --cacheinfo 160000,e21dc4bd5474d04b8e62d7331362edcc5648d7e5,tpls/llvm

The next step is replacing every occurrence of `llvm19`, `19_1_6`, and `19.1.6`
in `README.md` and all `docs/*.md` files. This command is useful for this task
(but make sure to check the results are right):

    $ git help sed
    'sed' is aliased to '! git ls-files -s -z | grep -z -v '^16' | cut -z -f 2- | xargs -0 sed -i -e'
    $ git sed 's/\<llvm19\>/llvmXX/g' -e 's/v19_1_6/vXX_Y_Z/g' -e 's/\<19_1_6\>/XX_Y_Z/g' -e 's/\<19\.1\.6/XX.Y.Z/g'

Then create a new release named "LLVM XX.Y.Z" and with the tag `vXX.Y.Z` here:
<https://github.com/gt-tinker/qwerty-llvm-builds/releases>. GitHub Actions
should build the tarballs automatically; you can track the progress here:
<https://github.com/gt-tinker/qwerty-llvm-builds/actions/workflows/build-llvm.yml>
