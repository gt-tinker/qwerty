# Horrifying MSVC Hack

This directory holds a ridiculous hack I implemented to work around what I
believe is a bug in `link.exe`, the Windows/MSVC linker.  If future versions of
MSVC resolve the issue, or future versions of CMake or Ninja work around it,
this hack should be removed immediately. (That would involve removing this
directory and the line that sets the `SKBUILD_BUILD_TOOL_ARGS` environment
variable in any CI scripts.)

## Symptom

The symptom of the issue is the following error when linking `qwerty-test`, the
GoogleTest binary for the Qwerty frontend unit tests:

    2025-06-27T02:47:19.0499625Z   LINK: command "C:\PROGRA~1\MICROS~2\2022\ENTERP~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\link.exe /nologo @CMakeFiles\qwerty-test.rsp /out:test\test_qwerty\qwerty-test.exe /implib:test\test_qwerty\qwerty-test.lib /pdb:test\test_qwerty\qwerty-test.pdb /version:0.0 /machine:x64 /STACK:10000000 /INCREMENTAL:NO /subsystem:console /MANIFEST:EMBED,ID=1" failed (exit code 1181) with the following output:
    2025-06-27T02:47:19.0502150Z   LINK : fatal error LNK1181: cannot open input file 'est_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_flag_immat.cpp.obj'

A glaring symptom of the problem is how `test_qwerty` appears to be truncated
to `est_qwerty` in the error message.

I have only observed this on GitHub-hosted GitHub Actions runners,
[specifically `windows-latest`][4]. Even with the same version of MSVC, I
cannot reproduce this locally.

## Likely Cause

In the command above, you'll notice that there is no list of object files and
libraries as you'd normally expect in a linker command line. That's because of
limitations on the `argv` size on Windows, apparently. Instead, the list of
things to link is found in the [_Response File_][1]
`CMakeFiles\qwerty-test.rsp`. Unfortunately, it can be difficult at first to
view this file to debug since Ninja deletes it by default, but thankfully,
Ninja has an undocumented flag `-d keeprsp` that keeps `.rsp` files around. (We
can get scikit-build-core to pass this flag by setting [`build.tool-args` in
`pyproject.toml`][2] to `["-d", "keeprsp"]`.)

After setting that flag, if you look at `_skbuild\CMakeFiles\qwerty-test.rsp`,
you'll see the following (I've abridged the last line by writing `[...]`
because it was gargantuan):

    test\test_qwerty\CMakeFiles\qwerty-test.dir\support\qwerty-test.cpp.obj
    test\test_qwerty\CMakeFiles\qwerty-test.dir\support\test_support.cpp.obj
    test\test_qwerty\CMakeFiles\qwerty-test.dir\test_ast.cpp.obj
    test\test_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_canonicalize.cpp.obj
    test\test_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_desugar.cpp.obj
    test\test_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_type_checking.cpp.obj
    test\test_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_flag_immat.cpp.obj  gtest-prefix\src\gtest-build\lib\gtest.lib  qwc.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMSupport.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMCore.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMX86CodeGen.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMX86AsmParser.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMX86Desc.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMX86Disassembler.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMX86Info.lib  D:\a\qwerty\qwerty\llvm20\lib\MLIRAffineAnalysis.lib [...] D:\a\qwerty\qwerty\llvm20\lib\LLVMMCParser.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMMC.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMIRReader.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMAsmParser.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMBitReader.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMCore.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMRemarks.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMBitstreamReader.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMTextAPI.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMBinaryFormat.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMTargetParser.lib  D:\a\qwerty\qwerty\llvm20\lib\LLVMSupport.lib  psapi.lib  shell32.lib  ole32.lib  uuid.lib  advapi32.lib  ws2_32.lib  ntdll.lib  delayimp.lib  -delayload:shell32.dll  -delayload:ole32.dll  D:\a\qwerty\qwerty\llvm20\lib\LLVMDemangle.lib  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib

Notice that the first few arguments to the linker are newline-delimited, and
the rest are space-delimited on the last line. It seems that this normally
works fine, but there may be a regression of some kind in MSVC that broke
support for either mixing both delimiters or just having very long lines. This
regression may have caused some memory corruption that broke the path of
`test_flag_immat.cpp.obj` as seen above.

### How the Response File is Generated

CMake generates a `build` edge in `build.ninja` like this (I've abridged with
`[...]` as above):

```ninja
build test\test_qwerty\qwerty-test.exe: CXX_EXECUTABLE_LINKER__qwerty-test_Release test\test_qwerty\CMakeFiles\qwerty-test.dir\support\qwerty-test.cpp.obj test\test_qwerty\CMakeFiles\qwerty-test.dir\support\test_support.cpp.obj test\test_qwerty\CMakeFiles\qwerty-test.dir\test_ast.cpp.obj test\test_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_canonicalize.cpp.obj test\test_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_desugar.cpp.obj test\test_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_type_checking.cpp.obj test\test_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_flag_immat.cpp.obj | gtest-prefix\src\gtest-build\lib\gtest.lib qwc.lib C$:\qwerty\llvm20\lib\LLVMSupport.lib C$:\qwerty\llvm20\lib\LLVMCore.lib C$:\qwerty\llvm20\lib\LLVMX86CodeGen.lib C$:\qwerty\llvm20\lib\LLVMX86AsmParser.lib C$:\qwerty\llvm20\lib\LLVMX86Desc.lib C$:\qwerty\llvm20\lib\LLVMX86Disassembler.lib C$:\qwerty\llvm20\lib\LLVMX86Info.lib C$:\qwerty\llvm20\lib\MLIRAffineAnalysis.lib [...] C$:\qwerty\llvm20\lib\LLVMMCParser.lib C$:\qwerty\llvm20\lib\LLVMMC.lib C$:\qwerty\llvm20\lib\LLVMIRReader.lib C$:\qwerty\llvm20\lib\LLVMAsmParser.lib C$:\qwerty\llvm20\lib\LLVMBitReader.lib C$:\qwerty\llvm20\lib\LLVMCore.lib C$:\qwerty\llvm20\lib\LLVMRemarks.lib C$:\qwerty\llvm20\lib\LLVMBitstreamReader.lib C$:\qwerty\llvm20\lib\LLVMTextAPI.lib C$:\qwerty\llvm20\lib\LLVMBinaryFormat.lib C$:\qwerty\llvm20\lib\LLVMTargetParser.lib C$:\qwerty\llvm20\lib\LLVMSupport.lib C$:\qwerty\llvm20\lib\LLVMDemangle.lib || lib\MLIRQCirc.lib lib\MLIRQCircTransforms.lib lib\MLIRQCircUtils.lib lib\MLIRQwerty.lib lib\MLIRQwertyAnalysis.lib lib\MLIRQwertyTransforms.lib lib\MLIRQwertyUtils.lib qwc.lib qwutil.lib
  FLAGS = /DWIN32 /D_WINDOWS   /Zc:inline /Zc:preprocessor /Zc:__cplusplus /Oi /bigobj /permissive- /W4 -wd4141 -wd4146 -wd4244 -wd4267 -wd4291 -wd4351 -wd4456 -wd4457 -wd4458 -wd4459 -wd4503 -wd4624 -wd4722 -wd4100 -wd4127 -wd4512 -wd4505 -wd4610 -wd4510 -wd4702 -wd4245 -wd4706 -wd4310 -wd4701 -wd4703 -wd4389 -wd4611 -wd4805 -wd4204 -wd4577 -wd4091 -wd4592 -wd4319 -wd4709 -wd5105 -wd4324 -wd4251 -wd4275 -w14062 -we4238 /Gw /O2 /Ob2  -MD
  LINK_FLAGS = /machine:x64 /STACK:10000000 /INCREMENTAL:NO /subsystem:console
  LINK_LIBRARIES = gtest-prefix\src\gtest-build\lib\gtest.lib  qwc.lib  C:\qwerty\llvm20\lib\LLVMSupport.lib  C:\qwerty\llvm20\lib\LLVMCore.lib  C:\qwerty\llvm20\lib\LLVMX86CodeGen.lib  C:\qwerty\llvm20\lib\LLVMX86AsmParser.lib  C:\qwerty\llvm20\lib\LLVMX86Desc.lib  C:\qwerty\llvm20\lib\LLVMX86Disassembler.lib  C:\qwerty\llvm20\lib\LLVMX86Info.lib  C:\qwerty\llvm20\lib\MLIRAffineAnalysis.lib  [...]  C:\qwerty\llvm20\lib\LLVMMCParser.lib  C:\qwerty\llvm20\lib\LLVMMC.lib  C:\qwerty\llvm20\lib\LLVMIRReader.lib  C:\qwerty\llvm20\lib\LLVMAsmParser.lib  C:\qwerty\llvm20\lib\LLVMBitReader.lib  C:\qwerty\llvm20\lib\LLVMCore.lib  C:\qwerty\llvm20\lib\LLVMRemarks.lib  C:\qwerty\llvm20\lib\LLVMBitstreamReader.lib  C:\qwerty\llvm20\lib\LLVMTextAPI.lib  C:\qwerty\llvm20\lib\LLVMBinaryFormat.lib  C:\qwerty\llvm20\lib\LLVMTargetParser.lib  C:\qwerty\llvm20\lib\LLVMSupport.lib  psapi.lib  shell32.lib  ole32.lib  uuid.lib  advapi32.lib  ws2_32.lib  ntdll.lib  delayimp.lib  -delayload:shell32.dll  -delayload:ole32.dll  C:\qwerty\llvm20\lib\LLVMDemangle.lib  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib
  OBJECT_DIR = test\test_qwerty\CMakeFiles\qwerty-test.dir
  POST_BUILD = cd .
  PRE_LINK = cd .
  TARGET_COMPILE_PDB = test\test_qwerty\CMakeFiles\qwerty-test.dir\
  TARGET_FILE = test\test_qwerty\qwerty-test.exe
  TARGET_IMPLIB = test\test_qwerty\qwerty-test.lib
  TARGET_PDB = test\test_qwerty\qwerty-test.pdb
  RSP_FILE = CMakeFiles\qwerty-test.rsp
```

This build edge invokes the following rule from `CMakeFiles\rules.ninja`:

```ninja
rule CXX_EXECUTABLE_LINKER__qwerty-test_Release
  command = C:\WINDOWS\system32\cmd.exe /C "$PRE_LINK && "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -E vs_link_exe --msvc-ver=1944 --intdir=$OBJECT_DIR --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100261~1.0\x64\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100261~1.0\x64\mt.exe --manifests $MANIFESTS -- C:\PROGRA~1\MIB055~1\2022\COMMUN~1\VC\Tools\MSVC\1444~1.352\bin\Hostx64\x64\link.exe /nologo @$RSP_FILE  /out:$TARGET_FILE /implib:$TARGET_IMPLIB /pdb:$TARGET_PDB /version:0.0 $LINK_FLAGS && $POST_BUILD"
  description = Linking CXX executable $TARGET_FILE
  rspfile = $RSP_FILE
  rspfile_content = $in_newline $LINK_PATH $LINK_LIBRARIES
  restat = $RESTAT
```

This rule is what is responsible for creating the `.rsp` file. It also explains
the inconsistency in delimiters: [the Ninja syntax `$in_newline`][3] is
hard-coded to put a newline between each prerequisite. Yet `$LINK_LIBRARIES`
is space-delimited as seen above.

## Workaround (Hack)

I wanted to generate an `.rsp` file with almost exclusively newlines, which I
suspect is more amenable to MSVC because `$in_newline` exists and was
tailor-made for MSVC. One idea would be modifying the `.rsp` file between Ninja
creating it and invoking `link.exe`, but I could not find an easy way to do
that. (I could have provided my own `link.exe` that modified response files
before passing them along, but that sounded difficult.) The only option that
comes to mind is modifying `build.ninja` after CMake runs, which is normally an
awful idea, but I have no better choice here given I am working with a broken
toolchain.

scikit-build-core is a blessing, but it complicates this, since it always runs
`cmake` followed by `ninja` every time you run `pip install .`. I am not aware
of any hooks to run code in between. After some thought, I resorted to using
the `build.tool-args` configuration flag for scikit-build-core to _choose a
different ninja file than `build.ninja`_ by passing the arguments `-f
shim.ninja` to Ninja. This shim ninja file (1) invokes a Python script that
modifies the real `build.ninja` and then (2) invokes the real `build.ninja` by
running `ninja`. The Python script changes each `build` block for each
executable in the real `build.ninja` to look like this instead of the earlier
example:

```ninja
build test\test_qwerty\qwerty-test.exe: CXX_EXECUTABLE_LINKER__qwerty-test_Release test\test_qwerty\CMakeFiles\qwerty-test.dir\support\qwerty-test.cpp.obj test\test_qwerty\CMakeFiles\qwerty-test.dir\support\test_support.cpp.obj test\test_qwerty\CMakeFiles\qwerty-test.dir\test_ast.cpp.obj test\test_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_canonicalize.cpp.obj test\test_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_desugar.cpp.obj test\test_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_type_checking.cpp.obj test\test_qwerty\CMakeFiles\qwerty-test.dir\ast_visitor\test_flag_immat.cpp.obj gtest-prefix\src\gtest-build\lib\gtest.lib qwc.lib C$:\qwerty\llvm20\lib\LLVMSupport.lib C$:\qwerty\llvm20\lib\LLVMCore.lib C$:\qwerty\llvm20\lib\LLVMX86CodeGen.lib C$:\qwerty\llvm20\lib\LLVMX86AsmParser.lib C$:\qwerty\llvm20\lib\LLVMX86Desc.lib C$:\qwerty\llvm20\lib\LLVMX86Disassembler.lib C$:\qwerty\llvm20\lib\LLVMX86Info.lib C$:\qwerty\llvm20\lib\MLIRAffineAnalysis.lib [...] C$:\qwerty\llvm20\lib\LLVMMCParser.lib C$:\qwerty\llvm20\lib\LLVMMC.lib C$:\qwerty\llvm20\lib\LLVMIRReader.lib C$:\qwerty\llvm20\lib\LLVMAsmParser.lib C$:\qwerty\llvm20\lib\LLVMBitReader.lib C$:\qwerty\llvm20\lib\LLVMCore.lib C$:\qwerty\llvm20\lib\LLVMRemarks.lib C$:\qwerty\llvm20\lib\LLVMBitstreamReader.lib C$:\qwerty\llvm20\lib\LLVMTextAPI.lib C$:\qwerty\llvm20\lib\LLVMBinaryFormat.lib C$:\qwerty\llvm20\lib\LLVMTargetParser.lib C$:\qwerty\llvm20\lib\LLVMSupport.lib C$:\qwerty\llvm20\lib\LLVMDemangle.lib
  FLAGS = /DWIN32 /D_WINDOWS   /Zc:inline /Zc:preprocessor /Zc:__cplusplus /Oi /bigobj /permissive- /W4 -wd4141 -wd4146 -wd4244 -wd4267 -wd4291 -wd4351 -wd4456 -wd4457 -wd4458 -wd4459 -wd4503 -wd4624 -wd4722 -wd4100 -wd4127 -wd4512 -wd4505 -wd4610 -wd4510 -wd4702 -wd4245 -wd4706 -wd4310 -wd4701 -wd4703 -wd4389 -wd4611 -wd4805 -wd4204 -wd4577 -wd4091 -wd4592 -wd4319 -wd4709 -wd5105 -wd4324 -wd4251 -wd4275 -w14062 -we4238 /Gw /O2 /Ob2  -MD
  LINK_FLAGS = /machine:x64 /STACK:10000000 /INCREMENTAL:NO /subsystem:console
  LINK_LIBRARIES = psapi.lib  shell32.lib  ole32.lib  uuid.lib  advapi32.lib  ws2_32.lib  ntdll.lib  delayimp.lib  -delayload:shell32.dll  -delayload:ole32.dll  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib
  OBJECT_DIR = test\test_qwerty\CMakeFiles\qwerty-test.dir
  POST_BUILD = cd .
  PRE_LINK = cd .
  TARGET_COMPILE_PDB = test\test_qwerty\CMakeFiles\qwerty-test.dir\
  TARGET_FILE = test\test_qwerty\qwerty-test.exe
  TARGET_IMPLIB = test\test_qwerty\qwerty-test.lib
  TARGET_PDB = test\test_qwerty\qwerty-test.pdb
  RSP_FILE = CMakeFiles\qwerty-test.rsp
```

Above, `|` has been removed (and everything following `||` was too because it
was redundant) and `LINK_LIBRARIES` only contains libraries that are not
dependencies. This means that the LLVM libraries are included in the
prerequisites that are passed to `$in_newline` instead of in `$LINK_LIBRARIES`.
There are still some space-delimited arguments in `$LINK_LIBRARIES`, but it
seems to be few enough that it fits in the tiny buffer inside `link.exe` that
we overflowed before.

I hope I can remove this hack as soon as possible. MSVC is not a serious tool.

[1]: https://learn.microsoft.com/en-us/visualstudio/msbuild/msbuild-response-files?view=vs-2022
[2]: https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html#configuring-cmake-arguments-and-defines
[3]: https://ninja-build.org/manual.html#ref_rule
[4]: https://github.com/actions/runner-images/tree/main#available-images
