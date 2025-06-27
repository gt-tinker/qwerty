@echo off
rem Once you've built LLVM, this script is useful for sourcing in cmd before
rem attemping a pip install of Qwerty

rem The /p flag sets $PATH and $MLIR_DIR too. Useful when building LLVM
rem locally, but probably not if you downloaded a prebuilt archive
if not "%1"=="/p" goto skipset

set PATH=%PATH%;C:\qwerty\build-installed\llvm20\bin
set MLIR_DIR=C:\qwerty\build-installed\llvm20\lib\cmake\mlir
:skipset

set CMAKE_GENERATOR=Ninja
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\vsdevcmd.bat" -no_logo -arch=x64
