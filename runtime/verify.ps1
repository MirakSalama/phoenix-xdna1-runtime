# Verify an already-installed release through Bionic and its native GGML scheduler.
[CmdletBinding()]
param(
    [string]$Results = (Join-Path (Get-Location).Path ('results\verify-' + (Get-Date -Format yyyyMMdd-HHmmss))),
    [string]$ModelRoot = (Join-Path $env:USERPROFILE '.lmstudio/models'),
    [switch]$IncludeLargeModels,
    [switch]$VerifyResidentBF16,
    [switch]$AllowEquivalentCityAnswer,
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
$plugin = Join-Path $installed 'ggml-phoenix.dll'
$payload = @{
    'llama-server.exe' = 'build\phoenix-server.exe'
    'ggml-phoenix.dll' = 'build\ggml-phoenix.dll'
}
foreach ($shape in $release.prefillPrograms) {
    foreach ($extension in @('bin','xclbin')) {
        $file = "programs\prefill-$shape.$extension"
        $payload[$file] = $file
    }
}
foreach ($shape in $release.residentPrograms) {
    foreach ($extension in @('bin','xclbin')) {
        $file = "programs\resident-$shape.$extension"
        $payload[$file] = $file
    }
}
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'programs') -Filter 'stream-*' -File) {
    $relative = "programs\$($file.Name)"
    $payload[$relative] = $relative
}
foreach ($file in $payload.Keys) {
    if ((Get-FileHash -LiteralPath (Join-Path $installed $file)).Hash -ne
        (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $payload[$file])).Hash) {
        throw "The installed Phoenix runtime differs from this release at $file."
    }
}
$smol = Join-Path $ModelRoot 'bartowski\SmolLM2-360M-Instruct-GGUF\SmolLM2-360M-Instruct-Q4_K_M.gguf'
$lfm = Join-Path $ModelRoot 'LiquidAI\LFM3-1.7B-GGUF\LFM3-1.7B-QAD-Q4_0.gguf'
$gemma = Join-Path $ModelRoot 'ggml-org\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_0.gguf'
$fixtures = [ordered]@{ q5_0 = $smol }
if ($IncludeLargeModels -or $VerifyResidentBF16) {
    $fixtures['q4_0-lfm3'] = $lfm
    $fixtures['q4_0-lfm3-down'] = $lfm
    $fixtures['q4_0-gemma'] = $gemma
    $fixtures['q4_0-gemma-wide'] = $gemma
    $fixtures['q4_0-gemma-down'] = $gemma
}
foreach ($model in $fixtures.Values | Sort-Object -Unique) {
    if (-not (Test-Path -LiteralPath $model -PathType Leaf)) { throw "Missing verification model: $model" }
}
& $program --phoenix-self-test *> (Join-Path $Results 'hardware.log')
if ($LASTEXITCODE) { throw "NPU self-test failed. See hardware.log in $Results." }
$previousPath = $env:PATH
try {
    $env:PATH = $cpu + ';' + $env:PATH
    # The native executable locates programs beside its own binary, just like the installed DLL.
    $scratch = Join-Path $Results ('phoenix-prefill-' + [guid]::NewGuid().ToString('N'))
    $programs = Join-Path $scratch 'programs'
    $null = [IO.Directory]::CreateDirectory($programs)
    $copied = @()
    try {
        $test = Join-Path $scratch 'prefill-regression.exe'
        $copied += $test
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'build\prefill-regression.exe') -Destination $test
        foreach ($source in Get-ChildItem -LiteralPath (Join-Path $installed 'programs') -Filter 'prefill-*' -File) {
            $destination = Join-Path $programs $source.Name
            $copied += $destination
            Copy-Item -LiteralPath $source.FullName -Destination $destination
        }
        & $test *> (Join-Path $Results 'prefill-regression.log')
        if ($LASTEXITCODE) { throw "Prefill activation/packing regression failed. See prefill-regression.log in $Results." }
    } finally {
        foreach ($item in $copied) { if ([IO.File]::Exists($item)) { [IO.File]::Delete($item) } }
        [IO.Directory]::Delete($programs)
        [IO.Directory]::Delete($scratch)
    }
    & (Join-Path $PSScriptRoot 'build\ggml-matvec-test.exe') $cpu (Join-Path $installed 'ggml-phoenix.dll') *> (Join-Path $Results 'numerics.log')
    if ($LASTEXITCODE) { throw "GGML numerical test failed. See numerics.log in $Results." }
    & (Join-Path $PSScriptRoot 'build\ggml-matvec-test.exe') $cpu (Join-Path $installed 'ggml-phoenix.dll') mapped *> (Join-Path $Results 'numerics-mapped.log')
    if ($LASTEXITCODE) { throw "Mapped-weight test failed. See numerics-mapped.log in $Results." }
    $numeric = Join-Path $PSScriptRoot 'build\ggml-prefill-test.exe'
    foreach ($fixture in $fixtures.Keys) {
        $tokenCounts = if ($fixture -eq 'q5_0') { @(97, 266) } else { @(129) }
        foreach ($tokens in $tokenCounts) {
            $log = Join-Path $Results "numerics-$fixture-$tokens.log"
            & $numeric $cpu $plugin $fixture $fixtures[$fixture] $tokens *> $log
            if ($LASTEXITCODE) { throw "GGUF projection test failed. See $log." }
        }
        if ($fixture -ne 'q5_0') {
            $log = Join-Path $Results "numerics-$fixture-canonical.log"
            & $numeric $cpu $plugin $fixture $fixtures[$fixture] 129 'cpu-buffer' *> $log
            if ($LASTEXITCODE) { throw "Canonical GGUF projection test failed. See $log." }
        }
    }
    if ($VerifyResidentBF16) {
        # Test real GGUFs directly in child processes without touching Bionic's stored mode.
        $savedResident = $env:PHOENIX_XDNA_RESIDENT_BF16
        $savedBudget = $env:PHOENIX_XDNA_PROMPT_CACHE_MIB
        try {
            $env:PHOENIX_XDNA_RESIDENT_BF16 = '1'
            $env:PHOENIX_XDNA_PROMPT_CACHE_MIB = $null
            foreach ($fixture in $fixtures.Keys | Where-Object { $_ -ne 'q5_0' }) {
                $storage = if ($fixture -eq 'q4_0-lfm3-down') { 'compact-q4' } else { 'resident-bf16' }
                foreach ($tokens in @(129,257)) {
                    foreach ($layout in @('canonical','repacked')) {
                        [string[]]$options = if ($layout -eq 'canonical') { @('cpu-buffer') } else { @() }
                        $log = Join-Path $Results "resident-$fixture-$tokens-$layout.log"
                        & $numeric $cpu $plugin $fixture $fixtures[$fixture] $tokens @options *> $log
                        if ($LASTEXITCODE -or !(Select-String -LiteralPath $log -SimpleMatch "storage=$storage " -Quiet)) {
                            throw "Resident regression expected $storage. Check available RAM and see $log."
                        }
                    }
                }
            }
            $env:PHOENIX_XDNA_RESIDENT_BF16 = '0'
            $log = Join-Path $Results 'resident-disabled.log'
            & $numeric $cpu $plugin 'q4_0-lfm3' $lfm 129 *> $log
            if ($LASTEXITCODE -or !(Select-String -LiteralPath $log -SimpleMatch 'storage=compact-q4 ' -Quiet)) {
                throw "Compact control failed. See $log."
            }
        } finally {
            $env:PHOENIX_XDNA_RESIDENT_BF16 = $savedResident
            $env:PHOENIX_XDNA_PROMPT_CACHE_MIB = $savedBudget
        }
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
    & python -B $harness $alias --placement split-head --workload $workload @request --reference $baseline @comparison --report (Join-Path $Results "phoenix-$workload.json")
    if ($LASTEXITCODE) { throw "Phoenix $workload benchmark failed." }
}
if ($IncludeLargeModels) {
    $models = @('--model', 'lfm3-1.7b', '--model', 'gemma-4-e2b-it', '--reasoning-off')
    foreach ($workload in @('count', 'prefill256', 'documents1000')) {
        $baseline = Join-Path $Results "cpu-large-$workload.json"
        $longPrompt = $workload -ne 'count'
        [string[]]$request = @()
        if ($longPrompt) { $request += '--full-answer' }
        if ($workload -eq 'documents1000') { $request += @('--context-length', '1536') }
        & python -B $harness cpu --workload $workload @models @request --report $baseline
        if ($LASTEXITCODE) { throw "CPU large-model $workload benchmark failed." }
        [string[]]$comparison = if ($longPrompt) {
            @('--enable-prefill', '--require-prefill-on', 'lfm3-1.7b', '--require-prefill-on', 'gemma-4-e2b-it')
        } else { @() }
        if ($AllowEquivalentCityAnswer -and $workload -eq 'prefill256') {
            $comparison += '--allow-equivalent-city-answer'
        }
        & python -B $harness $alias --placement split-head --workload $workload @models @request @comparison --reference $baseline --report (Join-Path $Results "phoenix-large-$workload.json")
        if ($LASTEXITCODE) { throw "Phoenix large-model $workload benchmark failed." }
    }
}
Write-Output "Verified $alias. Results are in $Results."
