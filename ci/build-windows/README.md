Building LLVM
=============

- Build on your own Windows machine with `./build-llvm.ps1`.
- Build using a Windows VM using Docker:
  1. Change the values for `WINDOWS_VM_USERNAME` and `WINDOWS_VM_PASSWORD` in `.env` to any username and password you would like.
  2. To start the VM, run:

         docker compose up -d

  3. Wait 15-30 minutes for the VM to finish automatically setting up.
     1. You can view the VM at any point (even during setup) in your web browser at http://localhost:8006.
  4. Once you are at the Windows desktop, open a terminal.
  5. To install dependencies, run:

         Set-ExecutionPolicy Bypass -Force
         \\host.lan\Data\setup-vm.ps1

  6. To build LLVM, specify an LLVM version in the following command (e.g. `-version 19.1.6`):

         \\host.lan\Data\build-llvm.ps1 -version <your_version_here>

  7. To shutdown the VM:
     1. Fastest way: Shutdown from within the Windows VM, then on the host machine, run `docker compose down`.
     2. Lazy way: On host machine, run `docker compose down`.

Building Qwerty
===============

To build Qwerty, you can source `setup-env.bat` to set up LLVM (and MSVC)
environment variables properly for you. Afterward, you just need to set up (and
activate) the virtual environment and you are off to the races.
