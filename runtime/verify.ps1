# Verify an already-installed release through Bionic and its native GGML scheduler.
[CmdletBinding()]
param(
    [string]$Results = (Join-Path $PSScriptRoot ('results\verify-' + (Get-Date -Format yyyyMMdd-HHmmss))),
    [string]$ModelRoot = (Join-Path $env:USERPROFILE '.lmstudio/models'),
    [ValidateSet('paris','count','prefill128','prefill256')]
    [string[]]$Workloads = @('paris','count','prefill128','prefill256')
)
$ErrorActionPreference = 'Stop'
$release = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'compatibility.json') | ConvertFrom-Json
$backends = Join-Path $env:USERPROFILE '.lmstudio\extensions\backends'
$installed = Join-Path $backends "$($release.pack)-$($release.version)"
$cpu = Join-Path $backends "$($release.basePack)-$($release.baseVersion)"
$alias = "$($release.pack)@$($release.version)"
$null = New-Item -ItemType Directory -Path $Results -ErrorAction Stop
$Results = (Resolve-Path -LiteralPath $Results).Path
& (Join-Path $PSScriptRoot 'install.ps1') -CheckOnly
$program = Join-Path $installed 'llama-server.exe'
& $program --phoenix-self-test *> (Join-Path $Results 'hardware.log')
if ($LASTEXITCODE) { throw "NPU self-test failed. See hardware.log in $Results." }
$previousPath = $env:PATH
try {
    $env:PATH = $cpu + ';' + $env:PATH
    & (Join-Path $PSScriptRoot 'build\ggml-matvec-test.exe') $cpu (Join-Path $installed 'ggml-phoenix.dll') *> (Join-Path $Results 'numerics.log')
    if ($LASTEXITCODE) { throw "GGML numerical test failed. See numerics.log in $Results." }
    & (Join-Path $PSScriptRoot 'build\ggml-matvec-test.exe') $cpu (Join-Path $installed 'ggml-phoenix.dll') mapped *> (Join-Path $Results 'numerics-mapped.log')
    if ($LASTEXITCODE) { throw "Mapped-weight test failed. See numerics-mapped.log in $Results." }
    $smol = Join-Path $ModelRoot 'bartowski\SmolLM2-360M-Instruct-GGUF\SmolLM2-360M-Instruct-Q4_K_M.gguf'
    foreach ($tokens in @(97, 266)) {
        & (Join-Path $PSScriptRoot 'build\ggml-prefill-test.exe') $cpu (Join-Path $installed 'ggml-phoenix.dll') q5_0 $smol $tokens *> (Join-Path $Results "numerics-prompt-$tokens.log")
        if ($LASTEXITCODE) { throw "GGUF prompt test failed at $tokens tokens. See the numerical log in $Results." }
    }
} finally { $env:PATH = $previousPath }
$harness = Join-Path $PSScriptRoot 'tests\bionic_performance.py'
foreach ($workload in $Workloads) {
    $baseline = Join-Path $Results "cpu-$workload.json"
    $longPrompt = $workload.StartsWith('prefill')
    [string[]]$request = if ($longPrompt) { @('--full-answer') } else { @() }
    & python -B $harness cpu --workload $workload @request --report $baseline
    if ($LASTEXITCODE) { throw "CPU $workload benchmark failed." }
    [string[]]$comparison = if ($longPrompt) { @('--enable-prefill', '--require-prefill-on', 'smollm2-360m-instruct') } else { @() }
    & python -B $harness $alias --workload $workload @request --reference $baseline @comparison --report (Join-Path $Results "phoenix-$workload.json")
    if ($LASTEXITCODE) { throw "Phoenix $workload benchmark failed." }
}
Write-Output "Verified $alias. Results are in $Results."
