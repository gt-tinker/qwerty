function Install-Visual-Studio-Community {
  Invoke-WebRequest -OutFile "vs_community.exe" "https://aka.ms/vs/17/release/vs_community.exe"
  Start-Process -FilePath "vs_community.exe" -ArgumentList '--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Component.VC.CMake.Project --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621'
}

function Install-Choco {
  Set-ExecutionPolicy Bypass -Scope Process -Force
  [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
  Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

function Install-Git {
  choco install -y git --ignore-package-exit-codes=3010
}

function Install-Python {
  choco install -y python312 --ignore-package-exit-codes=3010
}

function Main {
  Install-Visual-Studio-Community
  Install-Choco
  Install-Git
  Install-Python
}

Main