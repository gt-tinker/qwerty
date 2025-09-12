Building LLVM
=============

You have three options for building LLVM on Windows with these scripts:

## Option 1: Building on Local Windows Machine

Run `.\build-llvm.ps1 -version 19.1.6` in PowerShell.

## Option 2: Building with an Ephemeral Self-Hosted GitHub Actions Runner on AWS CodeBuild

1. Go into the AWS console and create a new ECR image repository. Update
   `build-docker-image.ps1` locally accordingly (if needed)
2. **On a local Windows machine,** run `.\build-docker-image.ps1`
3. In the AWS console under CodeBuild, create a new build project
   * Use GitHub as the source provider. It is crucial to set the "Connection"
     in the "Source" section to be a GitHub App _at the organization level_,
     **not** a GitHub App hooked up to your personal account.
   * Still under "Source," choose a "GitHub scoped webhook"
   * Under "Primary source webhook events," add only the `WORKFLOW_JOB_QUEUED`
     event type. **I had to [create my GitHub webhook][1] by hand using the
     option for that at creation time,** but you may not have to do that.
     (Let it try to register it for you first, I guess.)
   * Under "Environment," pick "Custom image," then "Windows 2019," and choose
     the ECR image repositorty from earlier. Choose "Project service role" to
     use the IAM role attached to this project to access ECR. **After setup,
     verify that the IAM role created has read access to your ECR image
     repository. I had to add these read permissions by hand**
   * Choose 15 GB of memory and 8 vCPUs under "Additional configuration" inside
     "Environment." More cores should speed up builds.
   * Ignore things like artifacts or batch configuration. The GitHub Actions
     runner will take care of things like that.
   * If creation fails, you will need to go to IAM and clean up the roles and
     polices created by the wizard. (The error handling by AWS is quite poor.)
4. Set up a GitHub Actions workflow like [the one in `qwerty-llvm-builds`][2]
   to run `.\build-llvm.ps1 -version 19.1.6` for you. This is the important
   part:
   ```
   runs-on: codebuild-qwerty-llvm-build-windows-${{ github.run_id }}-${{ github.run_attempt }}
   ```
   Note that `qwerty-llvm-build-windows` here is the name of the CodeBuild
   build project.

## Option 3: Building Using a Windows VM running with QEMU inside Docker:

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

To build Qwerty, you can run `.\setup-env.bat /p` to set up LLVM and MSVC
environment variables properly for you. Afterward, you just need to set up (and
activate) the virtual environment and you are off to the races.

[1]: https://docs.aws.amazon.com/codebuild/latest/userguide/github-manual-webhook.html
[2]: https://github.com/gt-tinker/qwerty-llvm-builds/blob/main/.github/workflows/build-llvm.yml
