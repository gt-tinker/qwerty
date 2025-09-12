# escape=`

# From:
# https://learn.microsoft.com/en-us/visualstudio/install/build-tools-container?view=vs-2022

# Use the latest Windows Server Core 2019 image.
# (Why 2019? That needs to match the version of the CodeBuild host, which is
# Windows Server 2019)
#FROM mcr.microsoft.com/windows/servercore:ltsc2019
# ...but Chocolately also needs .NET Framework 4.8, so use a base image that
# ships with .NET Framework 4.8
FROM mcr.microsoft.com/dotnet/framework/runtime:4.8-windowsservercore-ltsc2019

# Install Chocolatey
RUN powershell -NoLogo -NoProfile -Command "`
    Set-ExecutionPolicy Bypass -Scope Process -Force; `
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1')); `
    choco install -y git --ignore-package-exit-codes=3010; `
    choco install -y python312 --ignore-package-exit-codes=3010; `
    "

# Restore the default Windows shell for correct batch processing.
SHELL ["cmd", "/S", "/C"]

RUN `
    # Download the Build Tools bootstrapper.
    curl -SL --output vs_buildtools.exe https://aka.ms/vs/17/release/vs_buildtools.exe `
    `
    && (start /w vs_buildtools.exe --quiet --wait --norestart --nocache `
        --installPath "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools" `
        --add Microsoft.VisualStudio.Component.VC.CMake.Project `
        --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        --add Microsoft.VisualStudio.Component.Windows10SDK.19041 `
        || IF "%ERRORLEVEL%"=="3010" EXIT 0) `
    `
    # Cleanup
    && del /q vs_buildtools.exe

# Needed by CodeBuild even though that requirement is not documented.
# "Customer obsession," huh?
RUN msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi /qn
