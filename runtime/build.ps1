[CmdletBinding()]param([string]$XrtSdk = $env:XRT_ROOT)
$ErrorActionPreference = 'Stop'
if (!$XrtSdk -or !(Test-Path -LiteralPath (Join-Path $XrtSdk 'lib/xrt_coreutil.lib') -PathType Leaf)) {
    throw 'Set XRT_ROOT or pass -XrtSdk with the installed XRT SDK directory.'
}
$lock = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'runtime-lock.json') -Raw | ConvertFrom-Json -AsHashtable
$base = Join-Path $env:USERPROFILE ".lmstudio/extensions/backends/$($lock.vulkan)"
$out = Join-Path $PSScriptRoot 'build'
$objects = Join-Path $out 'objects'
$null = [IO.Directory]::CreateDirectory($objects)
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$dev = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'Common7/Tools/Launch-VsDevShell.ps1' | Select-Object -First 1
if (!$dev) { throw 'A Visual Studio C++ toolchain is required to rebuild the launcher.' }
Push-Location
try {
    & $dev -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
    Set-Location -LiteralPath $objects
    function Import-Library($name, $pattern) {
        $dll = Join-Path $base "$name.dll"
        if ((Get-FileHash -LiteralPath $dll).Hash -ne $lock.vulkanFiles["$name.dll"]) {throw "Unexpected pinned runtime: $name.dll"}
        $exports = & dumpbin.exe /nologo /exports $dll
        if ($LASTEXITCODE) {throw "Failed to inspect $name"}
        $symbols = @($exports | ForEach-Object {if ($_ -match "^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+($pattern)(?:\s|$)") {$Matches[1]}})
        if (!$symbols.Count) {throw "Missing expected exports in $name"}
        @("LIBRARY $name.dll",'EXPORTS')+$symbols | Set-Content -Encoding ascii -LiteralPath "$name.def"
        & lib.exe /nologo /machine:x64 "/def:$name.def" "/out:$name.lib" | Out-Null
        if ($LASTEXITCODE) {throw "Failed to create import library for $name"}
    }
    Import-Library 'llama-server-impl' '\?llama_server@@YAHHPEAPEAD@Z'
    Import-Library 'ggml_llamacpp' 'ggml_[A-Za-z0-9_]+'
    Import-Library 'ggml-base' '(?:(?:ggml|gguf)_[A-Za-z0-9_]+|dequantize_row_q8_K)'
    Import-Library 'ggml-cpu' 'ggml_[A-Za-z0-9_]+'
    $inc = "$PSScriptRoot/vendor/ggml/include"
    $flags = @('/nologo','/MD','/O2','/EHsc','/std:c++17','/utf-8','/Zc:__cplusplus','/DNDEBUG','/DGGML_SHARED',"/I$inc")
    & cl.exe @flags "$PSScriptRoot/src/hybrid.cpp" /link llama-server-impl.lib ggml_llamacpp.lib ggml-base.lib dxcore.lib "/out:$out/phoenix-hybrid.exe"
    if ($LASTEXITCODE) {throw 'Hybrid launcher compilation failed'}
    $sources = @('ggml-phoenix.cpp','phoenix_matvec.cpp','phoenix_prefill.cpp','self_test.cpp') |
        ForEach-Object {Join-Path "$PSScriptRoot/src" $_}
    & cl.exe @flags /LD /DGGML_BACKEND_SHARED /DGGML_BACKEND_DL /DGGML_BACKEND_BUILD "/I$PSScriptRoot/vendor/ggml/src" "/I$XrtSdk/include" @sources /link ggml-base.lib ggml-cpu.lib ggml_llamacpp.lib "$XrtSdk/lib/xrt_coreutil.lib" "/implib:$objects/ggml-phoenix.lib" "/out:$out/ggml-phoenix.dll"
    if ($LASTEXITCODE) {throw 'Phoenix backend compilation failed'}
} finally { Pop-Location }
