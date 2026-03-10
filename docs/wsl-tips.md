Tips for Building on WSL
========================

If you are working on Windows, we recommend using [WSL][1]. WSL allows
developers to run a Linux distro on a Windows machine without dealing with dual
boot or a virtual machine. Additionally, both WSL and Windows can interact with
each other's file systems. The reason we recommend WSL is that our support and
testing on Linux is far more extensive than on Windows.

First, **install WSL** by following this guide on [installing WSL][2].

Once you have WSL installed, you may follow this guide and others as if you
were on a Linux distro, with a couple of suggestions.

WSL Filesystem versus Windows Filesystem
----------------------------------------

Earlier we stated that WSL and Windows have access to each other file systems.
This is 100% true, with the caveat that it is [far, far slower][3] than
interacting with either's native file system. Microsoft admits as much in their
comparison between [WSL1 and WSL2][4].

The takeaway here is to clone and keep all Qwerty compiler development files in
the WSL file system (e.g., your homedir `~/` in WSL). As a real world example,
an undergraduate student cloned this repository using WSL onto their Windows
filesystem, where running a build was a multiday (>8 hrs) task. When the Qwerty
compiler source code was moved to the WSL filesystem, the build took at most 10
minutes.

WSL Ram Usage
-------------

By default, WSL is allocated **half** of the machine's RAM. If you'd like to
increase this limit, [this guide][5] is a nice and simple walkthrough.

The main symptom of this issue is your compilation process keeps getting sniped
by the OOM killer as seen below:

    c++: fatal error: Killed signal terminated program cc1plus
      compilation terminated.

Then a potential fix is to add the following code near the top of your
`CMakeLists.txt` both in `/` and  `/tweedledum`:

    set_property(GLOBAL APPEND PROPERTY JOB_POOLS link_job_pool=1)
    set(CMAKE_JOB_POOL_LINK link_job_pool)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS compile_job_pool=4)
    set(CMAKE_JOB_POOL_COMPILE compile_job_pool)

This tells CMake to tell Ninja to limit the number of linking and compilation
jobs done in parallel to 1 and 4 respectively, although this can be changed by
changing the above parameters.

Additional Troubleshooting
--------------------------

For more general issues with WSL, here is the [official troubleshooting
guide][6].

[1]: https://learn.microsoft.com/en-us/windows/wsl/about
[2]: https://learn.microsoft.com/en-us/windows/wsl/install
[3]: https://learn.microsoft.com/en-us/windows/wsl/filesystems
[4]: https://learn.microsoft.com/en-us/windows/wsl/compare-versions
[5]: https://fizzylogic.nl/2023/01/05/how-to-configure-memory-limits-in-wsl2
[6]: https://github.com/MicrosoftDocs/wsl/blob/main/WSL/troubleshooting.md
