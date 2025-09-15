# Usage:
# .\build-llvm.ps1 -version 21.1.1

param(
    [Parameter(Mandatory=$true)][string]$version
)

$ErrorActionPreference = "Stop"

if ($Env:GITHUB_WORKSPACE) {
    $root = $Env:GITHUB_WORKSPACE
} else {
    $root = "C:\qwerty"
}

$visual_studio_install_path = 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise'

if (-not (Test-Path $visual_studio_install_path)) {
    $visual_studio_install_path = 'C:\Program Files\Microsoft Visual Studio\2022\Community'
}
if (-not (Test-Path $visual_studio_install_path)) {
    $visual_studio_install_path = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
}
if (-not (Test-Path $visual_studio_install_path)) {
    Write-Error "Visual Studio install dir does not exist and I'm out of ideas: $visual_studio_install_path"
    exit 1
}

$llvm_version = $version
$llvm_dir = "$root\llvm"
$llvm_build_dir = "$root\build"
$llvm_major_version = $llvm_version.split('.')[0]
$llvm_install_dir = "$root\build-installed\llvm$llvm_major_version"

$starting_dir = $PWD.Path
$zipped_build_dirpath = "$root\build-zipped-up\llvm"
$llvm_version_underscores = $llvm_version.replace('.', '_')
$zipped_build_filepath = "$zipped_build_dirpath\llvm_mlir_rel_v${llvm_version_underscores}_x86_windows.zip"

echo "===========> Setting up Visual Studio environment variables. Wish me luck..."

# From: https://github.com/microsoft/vswhere/issues/150#issuecomment-485381959
if ($visual_studio_install_path -and (Test-Path "$visual_studio_install_path\Common7\Tools\vsdevcmd.bat")) {
  $json = $(& "${env:COMSPEC}" /s /c "`"$visual_studio_install_path\Common7\Tools\vsdevcmd.bat`" -no_logo -arch=x64 && powershell -Command `"Get-ChildItem env: | Select-Object Key,Value | ConvertTo-Json`"")
  if ($LASTEXITCODE -ne 0) {
    Write-Error "($LASTEXITCODE) $visual_studio_install_path\Common7\Tools\vsdevcmd.bat: $json"
  } else {
    # Write-Host $json
    $($json | ConvertFrom-Json) | ForEach-Object {
      $k, $v = $_.Key, $_.Value
      Set-Content env:\"$k" "$v"
    }
  }
}

If(!(test-path -Path $llvm_dir)) {
    echo "===========> LLVM repository path $llvm_dir does not exist, cloning..."
    git clone "https://github.com/llvm/llvm-project.git" $llvm_dir
} Else {
    echo "===========> LLVM repository path $llvm_dir already exists"
}

echo "===========> Pulling and checking out version $llvm_version..."

git config --global --add safe.directory $llvm_dir
cd $llvm_dir
git fetch origin --tags
git checkout "llvmorg-$llvm_version"

echo "===========> Configuring..."

cd $llvm_dir
# Turning off DIA with `-DLLVM_ENABLE_DIA_SDK=OFF` is more important than it
# looks; without it, the generated LLVM CMake scripts end up making the Qwerty
# CMake scripts attempt to link with `C:/Program Files/Microsoft Visual
# Studio/2022/Community/DIA SDK/lib/amd64/diaguids.lib`, which may not exist on
# some systems, namely GitHub Actions hosts.
cmake -S llvm -B "$llvm_build_dir" -DLLVM_ENABLE_PROJECTS=mlir -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=TRUE "-DCMAKE_INSTALL_PREFIX=$llvm_install_dir" -DLLVM_TARGETS_TO_BUILD=Native -DLLVM_ENABLE_DIA_SDK=OFF -Thost=x64 -DLLVM_INSTALL_UTILS=TRUE

echo "===========> Building $llvm_build_dir\LLVM.sln (be patient)..."

cd $llvm_build_dir
msbuild ALL_BUILD.vcxproj -p:Configuration=Release

echo "===========> Installing to $llvm_install_dir..."
msbuild INSTALL.vcxproj -p:Configuration=Release

echo "===========> Did it work? This should print out version $llvm_version"
cd $llvm_install_dir
bin\llvm-config --version

echo "===========> Zipping up build to $zipped_build_filepath..."
New-Item -ItemType Directory -Force -Path $zipped_build_dirpath
Compress-Archive -Path $llvm_install_dir -CompressionLevel Fastest -DestinationPath $zipped_build_filepath
echo "===========> Zipped up build to $zipped_build_filepath."

cd $starting_dir
