Upgrading LLVM
==============

Updating LLVM is not the most fun thing in the world, but on the bright side,
it often brings bugfixes and improved APIs. The first step of updating the LLVM
dependency is to update the LLVM submodule we use to track the current LLVM commit
we're compatible with (CIRCT does something similar):

    $ git update-index --cacheinfo 160000,e21dc4bd5474d04b8e62d7331362edcc5648d7e5,tpls/llvm

The next step is replacing every occurrence of `llvm20`, `20_1_7`, and `20.1.7`
in `README.md` and all `docs/*.md` files. This command is useful for this task
(but make sure to check the results are right):

    $ git sed 's/\<llvm20\>/llvmXX/g ; s/v20_1_7/vXX_Y_Z/g ; s/\<20_1_7\>/XX_Y_Z/g ; s/20\.1\.6/XX.Y.Z/g'

The git sed command used above is a custom alias. It's not a standard Git command and may not work out of the box on a random machine. To use it, you need to define the alias yourself. Here’s how to set it up:

For macOS:

    $ git config --global alias.sed '! git ls-files -s | grep -v '\''^16'\'' | cut -f 2- | xargs env LC_ALL=C sed -E -i '\'\'' -e'
    
For Linux:

    $ git config --global alias.sed '! git ls-files -s -z | grep -z -v '\''^16'\'' | cut -z -f 2- | xargs -0 sed -i -e'

Then create a new release named "LLVM XX.Y.Z" and with the tag `vXX.Y.Z` here:
<https://github.com/gt-tinker/qwerty-llvm-builds/releases>. GitHub Actions
should build the tarballs automatically; you can track the progress here:
<https://github.com/gt-tinker/qwerty-llvm-builds/actions/workflows/build-llvm.yml>
