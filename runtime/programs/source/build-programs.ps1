# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
[CmdletBinding()]
param([string]$Toolchain = $env:PHOENIX_IRON_ROOT, [string]$Cache, [string]$Output,
      [ValidateSet('paired16', 'vector8', 'scalar8')][string]$Variant = 'paired16')
$ErrorActionPreference='Stop'
if (!$Toolchain) {throw 'Set PHOENIX_IRON_ROOT or pass -Toolchain pointing to the IRON toolchain tree.'}
if (!$Output) {$Output=Join-Path $PSScriptRoot "rebuilt/$Variant"}
if (!$Cache) {$Cache=Join-Path $PSScriptRoot 'build/iron-cache'}
$python=Join-Path $Toolchain 'python313'
$site=Join-Path $Toolchain 'mlir-aie/ironenv/Lib/site-packages'
$env:MLIR_AIE_INSTALL_DIR="$site/mlir_aie"
$env:PEANO_INSTALL_DIR="$site/llvm-aie"
$env:XRT_ROOT="$Toolchain/xrt_sdk/xrt"
$env:PYTHONPATH="$site;$site/mlir_aie/python;$env:XRT_ROOT/python"
$env:PATH="$python;$python/Library/bin;$env:MLIR_AIE_INSTALL_DIR/bin;$env:MLIR_AIE_INSTALL_DIR/lib;$env:XRT_ROOT;$env:XRT_ROOT/lib;$env:SystemRoot/System32/AMD;$env:PATH"
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NPU_CACHE_HOME=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Cache)
$env:NPU2='0'
$vswhere=Join-Path ([Environment]::GetFolderPath('ProgramFilesX86')) 'Microsoft Visual Studio/Installer/vswhere.exe'
$devShell=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'Common7/Tools/Launch-VsDevShell.ps1' | Select-Object -First 1
if (!$devShell) {throw 'Visual Studio C++ Build Tools is required.'}
& $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
[string[]]$features = @()
if ($Variant -ne 'scalar8') {$features += '--vector-scale'}
if ($Variant -eq 'paired16') {$features += '--paired'}
& "$python/python.exe" "$PSScriptRoot/bonsai_q1.py" $Output @features
exit $LASTEXITCODE
