param([string]$Toolchain = $env:PHOENIX_IRON_ROOT,
      [string]$Cache, [string]$Output)
$ErrorActionPreference = 'Stop'
if (!$Toolchain) {throw 'Supply -Toolchain or set PHOENIX_IRON_ROOT to the IRON toolchain directory.'}
$pack = (Resolve-Path "$PSScriptRoot/../..").Path
if (!$Cache) { $Cache = Join-Path $pack 'build/iron-cache' }
if (!$Output) { $Output = Join-Path $pack 'programs' }
$python = Join-Path $Toolchain 'python313'
$site = Join-Path $Toolchain 'mlir-aie/ironenv/Lib/site-packages'
$env:MLIR_AIE_INSTALL_DIR = "$site/mlir_aie"
$env:PEANO_INSTALL_DIR = "$site/llvm-aie"
$env:XRT_ROOT = "$Toolchain/xrt_sdk/xrt"
$env:PYTHONPATH = "$site;$site/mlir_aie/python;$env:XRT_ROOT/python"
$env:PATH = "$python;$python/Library/bin;$env:MLIR_AIE_INSTALL_DIR/bin;$env:MLIR_AIE_INSTALL_DIR/lib;$env:XRT_ROOT;$env:XRT_ROOT/lib;$env:SystemRoot/System32/AMD;$env:PATH"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NPU_CACHE_HOME = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Cache)
$env:NPU2 = '0'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$devShell = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'Common7/Tools/Launch-VsDevShell.ps1' | Select-Object -First 1
if (!$devShell) { throw 'Visual Studio with the C++ build tools is required.' }
& $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
& "$python/python.exe" "$PSScriptRoot/stream_gemv.py" $Output
if ($LASTEXITCODE) { throw "GEMV program build failed with exit code $LASTEXITCODE." }
& "$python/python.exe" "$PSScriptRoot/prefill_gemm.py" $Output
if ($LASTEXITCODE) { throw "Prefill GEMM program build failed with exit code $LASTEXITCODE." }
& "$python/python.exe" "$PSScriptRoot/prefill_gemm.py" $Output --tokens 256
if ($LASTEXITCODE) { throw "Wide prefill GEMM program build failed with exit code $LASTEXITCODE." }
foreach ($tokens in @(128,256)) {
    & "$python/python.exe" "$PSScriptRoot/prefill_gemm.py" $Output --tokens $tokens --resident
    if ($LASTEXITCODE) { throw "Resident BF16 program build failed at $tokens tokens." }
}
