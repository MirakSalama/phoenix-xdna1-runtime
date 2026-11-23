[CmdletBinding()]
param([Parameter(Mandatory)][ValidateSet('cpu-npu','cpu-prompt','cpu-prompt-resident','radeon','split-head','split-head-bf16','prompt-ffn')][string]$Mode)
$ErrorActionPreference='Stop'
$lock=Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'runtime-lock.json') | ConvertFrom-Json
$base=(Resolve-Path -LiteralPath (Join-Path $env:USERPROFILE '.lmstudio/extensions/backends')).Path
$pack=(Resolve-Path -LiteralPath (Join-Path $base "$($lock.name)-$($lock.version)")).Path
$manifest=Get-Content -Raw -LiteralPath (Join-Path $pack 'backend-manifest.json') | ConvertFrom-Json
if ([IO.Path]::GetDirectoryName($pack) -ne $base -or $manifest.name -ne $lock.name -or
    $manifest.version -ne $lock.version -or ((Get-Item -LiteralPath $pack).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw 'Unexpected experimental runtime directory'
}
$path=Join-Path $pack 'hybrid-mode.txt'
$temp=Join-Path $pack ('.hybrid-mode-'+[guid]::NewGuid().ToString('N'))
try {
    [IO.File]::WriteAllText($temp,"$Mode`n",[Text.UTF8Encoding]::new($false))
    [IO.File]::Move($temp,$path,$true)
} finally {
    if (Test-Path -LiteralPath $temp) {Remove-Item -LiteralPath $temp}
}
Write-Output "Hybrid mode set to $Mode. Unload and reload the model in Bionic to apply it."
