# Configure the installed version. Each new Bionic model instance reads the setting at startup.
[CmdletBinding(DefaultParameterSetName='default')]
param(
    [Parameter(Position=0)][ValidateSet('split-head','prompt-only')][string]$Mode,
    [Parameter(ParameterSetName='resident')][switch]$ResidentBF16,
    [Parameter(ParameterSetName='compact')][switch]$Compact
)
$ErrorActionPreference = 'Stop'
if (!$Mode -and !$ResidentBF16 -and !$Compact) { throw 'Choose -Mode, -ResidentBF16, or -Compact.' }
$release = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'compatibility.json') -Raw | ConvertFrom-Json
$backends = (Resolve-Path -LiteralPath (Join-Path $env:USERPROFILE '.lmstudio/extensions/backends')).Path
$pack = Join-Path $backends "$($release.pack)-$($release.version)"
$installed = Get-Content -LiteralPath (Join-Path $pack 'backend-manifest.json') -Raw | ConvertFrom-Json
if ($installed.name -ne $release.pack -or $installed.version -ne $release.version) { throw 'Installed version mismatch.' }
function Set-PackMarker([string]$name, [bool]$enabled) {
    $marker = Join-Path $pack $name
    if (Test-Path -LiteralPath $marker) {
        $item = Get-Item -Force -LiteralPath $marker
        if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $item.Length -ne 0) {
            throw "Expected an empty regular file at $marker."
        }
        if (!$enabled) { [IO.File]::Delete($marker) }
    } elseif ($enabled) {
        [IO.File]::Open($marker, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read).Dispose()
    }
}
if ($Mode) { Set-PackMarker 'phoenix-prompt-only' ($Mode -eq 'prompt-only') }
if ($ResidentBF16 -or $Compact) { Set-PackMarker 'phoenix-resident-bf16' ([bool]$ResidentBF16) }
$placement = if ([IO.File]::Exists((Join-Path $pack 'phoenix-prompt-only'))) { 'prompt-only' } else { 'split-head' }
$storage = if ([IO.File]::Exists((Join-Path $pack 'phoenix-resident-bf16'))) { 'resident BF16 when memory permits' } else { 'compact' }
Write-Output "Phoenix is configured for $placement with $storage weights. Reload the model in Bionic to apply it."
