Building LLVM Yourself
======================

First, you need to clone [the LLVM repository][5] and check out the
`llvmorg-19.1.6` tag. For example, the following commands would check out the
right commit (I would _strongly_ recommend running these outside this
repository):

    $ git clone https://github.com/llvm/llvm-project.git llvm
    $ cd llvm
    $ git checkout llvmorg-19.1.6

There are some scripts for building LLVM on different operating systems under
`docs/scripts/`, but the instructions below guide you through building LLVM by
hand.

macOS/Linux
-----------

Try the following. Say `Release` instead of `RelWithDebInfo` if you don't need
debug symbols and want to save some disk space. You should also set
`-DLLVM_PARALLEL_LINK_JOBS=1` instead of 2 if you have less than ~32 GB of RAM.
You may want to leave out enabling assertions too, if you care about the
performance impact, but this may cause you other problems — see the
[Debugging guide](debugging.md) of the README). Also, on macOS,
you don't need to bother specifying `-DLLVM_USE_LINKER=gold` since `lld` is
fast enough (the GNU `ld` installed on a typical GNU/Linux system is the slow
one). Also, `-DLLVM_INSTALL_UTILS=TRUE` is important if you want to run the
test suite, since it installs the LLVM [FileCheck][4] tool we use in testing.

    $ mkdir build && cd build
    $ cmake -G Ninja -DLLVM_ENABLE_PROJECTS=mlir -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_ASSERTIONS=TRUE -DLLVM_PARALLEL_LINK_JOBS=2 -DCMAKE_INSTALL_PREFIX=/path_to_llvm_install_dir/ -DLLVM_TARGETS_TO_BUILD=Native -DLLVM_USE_LINKER=gold -DLLVM_INSTALL_UTILS=TRUE ../llvm
    $ ninja install

You'll need to do the following as well:

    $ export MLIR_DIR=/path_to_llvm_install_dir/lib/cmake/mlir/
    $ export PATH=$PATH:/path_to_llvm_install_dir/bin/

You will probably want to put both `export`s in [your `~/.bashrc`][1].

Windows
-------

I had luck following [the LLVM guide for LLVM in Visual Studio][2]. Here is
some Qwerty-specific advice to complement their guide:

1. Use the `git clone` command above with `C:\qwerty` as your current directory
   (that is, clone LLVM at `C:\qwerty\llvm-project`). Check out the tag
   `llvmorg-19.1.6`.
2. When you install Python, if you want to do a debug build of the Qwerty
   compiler, you should check "Download debugging symbols" and "Downloaded debug
   binaries" under "Advanced Options" in the installer. This will install
   `python312_d.lib` (as opposed to only `python312.lib` ), without which you
   may see some linker errors when you do a debug build. _(Note, however, that
   debug builds are currently broken on Windows due to other linker problems.)_
3. I used the following `cmake` command line **inside an _administrator_ "x64
   Native Tools Command Prompt for VS 2022"**:
   ```
   cmake -S llvm\llvm -B build -DLLVM_ENABLE_PROJECTS=mlir -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=TRUE -DCMAKE_INSTALL_PREFIX=C:\qwerty\llvm19 -DLLVM_TARGETS_TO_BUILD=Native -DLLVM_ENABLE_DIA_SDK=OFF -Thost=x64 -DLLVM_INSTALL_UTILS=TRUE
   ```
   (Turning off [DIA][3] with `-DLLVM_ENABLE_DIA_SDK=OFF` is more important
   than it looks; without it, the generated LLVM CMake scripts end up making
   the Qwerty CMake scripts attempt to link with
   `C:/Program Files/Microsoft Visual Studio/2022/Community/DIA SDK/lib/amd64/diaguids.lib`,
   which may not exist on some systems, namely GitHub Actions hosts.)
4. The guide mentions this, but make sure that when you open `C:\qwerty\llvm\`,
   you choose "Release" as your configuration here:\
   ![Screenshot of "Release" as the Visual Studio Configuration][img:vs-config]
5. To build, you can right-click "ALL_BUILD" under
   "CMakePredefinedTargets" in Solution Explorer and choose "Build":\
   ![Screenshot of "ALL_BUILD" inside "CMakePredefinedTargets"][img:vs-project]\
   This took about an hour on my machine, during which it was otherwise unusable.
6. To install, you can right-click "INSTALL" in the same place and
   again choose "Build". If you skip the previous step, this fails (in my
   experience).
7. Press _Start_ and type in "environment variables" (I would recommend editing
   system variables, not user variables).
   Set `MLIR_DIR` to `C:\qwerty\llvm19\lib\cmake\mlir` and add
   `C:\qwerty\llvm19\bin` to `Path`

[1]: https://unix.stackexchange.com/q/129143/62375
[2]: https://llvm.org/docs/GettingStartedVS.html
[3]: https://learn.microsoft.com/en-us/visualstudio/debugger/debug-interface-access/debug-interface-access-sdk?view=vs-2022
[4]: https://llvm.org/docs/CommandGuide/FileCheck.html
[5]: https://github.com/llvm/llvm-project

[img:vs-config]: img/vs-config.png
[img:vs-project]: img/vs-project.png
