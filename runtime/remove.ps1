# Remove the pack from Bionic's runtime list, preserving its contents and diagnostics.
[CmdletBinding()]param()
$ErrorActionPreference='Stop'
$lock=Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'runtime-lock.json') | ConvertFrom-Json
$lmsroot=(Resolve-Path -LiteralPath (Join-Path $env:USERPROFILE '.lmstudio')).Path
$backends=(Resolve-Path -LiteralPath (Join-Path $lmsroot 'extensions/backends')).Path
$expected="$($lock.name)-$($lock.version)"
if ($expected -notin 'phoenix-bonsai-preview-win-x86_64-avx2-0.5.0','phoenix-xdna1-win-x86_64-avx2-0.5.0') {
    throw 'Unexpected runtime identity'
}
$src=Join-Path $backends $expected
if (!(Test-Path -LiteralPath $src)) {Write-Output 'The experimental runtime is already absent.'; return}
$src=(Resolve-Path -LiteralPath $src).Path
$manifest=Get-Content -Raw -LiteralPath (Join-Path $src 'backend-manifest.json') | ConvertFrom-Json
if ((Get-FileHash -LiteralPath (Join-Path $src 'llama-server.exe')).Hash -ne $lock.payloadFiles.'llama-server.exe' -or
    (Get-FileHash -LiteralPath (Join-Path $src 'ggml-phoenix.dll')).Hash -ne $lock.payloadFiles.'ggml-phoenix.dll') {
    throw 'The installed native binaries differ from this package. Use its matching removal script.'
}
if ([IO.Path]::GetDirectoryName($src) -ne $backends -or [IO.Path]::GetFileName($src) -ne $expected -or
    $manifest.name -ne $lock.name -or $manifest.version -ne $lock.version -or
    ((Get-Item -LiteralPath $src).Attributes -band [IO.FileAttributes]::ReparsePoint)) {throw 'Refusing to touch a different or linked runtime.'}
if (Get-Process -Name llama-server -ErrorAction SilentlyContinue | Where-Object {$_.Path -eq "$src\llama-server.exe"}) {
    throw 'Unload the model using this experimental runtime before removal.'
}
$cli=Join-Path $lmsroot 'bin/lms.exe'
$listing=& $cli runtime ls 2>&1
if ($LASTEXITCODE) {throw 'Bionic runtime inventory unavailable.'}
if ($listing -match ([regex]::Escape("$($lock.name)@$($lock.version)")+".*"+[char]0x2713)) {
    throw 'Select another GGUF engine in Bionic before removal.'
}
$archive=Join-Path $lmsroot 'retired-hybrid-runtimes'
$null=[IO.Directory]::CreateDirectory($archive)
if ((Resolve-Path -LiteralPath $archive).Path -ne $archive -or
    ((Get-Item -LiteralPath $archive).Attributes -band [IO.FileAttributes]::ReparsePoint)) {throw 'Unexpected archive path'}
$dest=Join-Path $archive ($expected+'-'+[guid]::NewGuid().ToString('N'))
[IO.Directory]::Move($src,$dest)
Write-Output "Removed experimental runtime from Bionic. Archived its files at $dest."
