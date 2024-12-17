@echo off
rem Once you've built LLVM, this script is useful for sourcing in cmd before
rem attemping a pip install of Qwerty

set CMAKE_GENERATOR=Ninja
set PATH=%PATH%;C:\qwerty\build-installed\llvm19\bin
set MLIR_DIR=C:\qwerty\build-installed\llvm19\lib\cmake\mlir

"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\vsdevcmd.bat" -no_logo -arch=x64
