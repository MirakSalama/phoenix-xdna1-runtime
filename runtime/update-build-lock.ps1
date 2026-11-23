[CmdletBinding()]param()
$ErrorActionPreference = 'Stop'
$path = Join-Path $PSScriptRoot 'runtime-lock.json'
$lock = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json -AsHashtable
$files = [ordered]@{
    'llama-server.exe' = 'build/phoenix-hybrid.exe'
    'ggml-phoenix.dll' = 'build/ggml-phoenix.dll'
    'programs/bonsai-q1-3072x5120.xclbin' = 'programs/bonsai-q1-3072x5120.xclbin'
    'programs/bonsai-q1-3072x5120.bin' = 'programs/bonsai-q1-3072x5120.bin'
}
$hashes = [ordered]@{}
foreach ($name in $files.Keys) {
    $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $files[$name])).Hash
}
$lock.payloadFiles = $hashes
$json = ($lock | ConvertTo-Json -Depth 8) + [Environment]::NewLine
$temporary = Join-Path $PSScriptRoot ('.payload-lock-' + [guid]::NewGuid().ToString('N'))
try {
    [IO.File]::WriteAllText($temporary,$json,[Text.UTF8Encoding]::new($false))
    [IO.File]::Move($temporary,$path,$true)
} finally {
    if (Test-Path -LiteralPath $temporary) {Remove-Item -LiteralPath $temporary}
}
