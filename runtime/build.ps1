param(
    [string]$BaseRuntime = (Join-Path $env:USERPROFILE '.lmstudio\extensions\backends\llama.cpp-win-x86_64-avx2-2.80.1'),
    [string]$XrtSdk = $env:XRT_ROOT,
    [switch]$LauncherOnly
)
$ErrorActionPreference = 'Stop'
if (!$XrtSdk -and !$LauncherOnly) { throw 'Supply -XrtSdk or set XRT_ROOT to the XRT SDK directory.' }
$root = $PSScriptRoot
$build = Join-Path $root 'build'
$objects = Join-Path $build 'objects'
$src = Join-Path $root 'src'
$inc = Join-Path $root 'vendor\ggml\include'
$internals = Join-Path $root 'vendor\ggml\src'
$null = New-Item -ItemType Directory -Force -Path $objects

$manifest = Get-Content -LiteralPath (Join-Path $BaseRuntime 'backend-manifest.json') -Raw | ConvertFrom-Json
if ($manifest.name -ne 'llama.cpp-win-x86_64-avx2' -or $manifest.version -ne '2.80.1') {
    throw 'The bundled GGML headers target the 2.80.1 CPU runtime (llama.cpp b12600).'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$devShell = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'Common7\Tools\Launch-VsDevShell.ps1' | Select-Object -First 1
if (-not $devShell) { throw 'Visual Studio C++ Build Tools and a Windows SDK are required.' }

function New-ImportLibrary([string]$libraryName, [string]$symbols) {
    $exports = & dumpbin.exe /nologo /exports (Join-Path $BaseRuntime "$libraryName.dll")
    if ($LASTEXITCODE) { throw "Reading exports of $libraryName failed." }
    $names = @($exports | ForEach-Object { if ($_ -match "^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+($symbols)(?:\s|$)") { $Matches[1] } })
    if ($names.Count -eq 0) { throw "Could not find the expected exports of $libraryName." }
    $definition = Join-Path $objects "$libraryName.def"
    @("LIBRARY $libraryName.dll", 'EXPORTS') + $names | Set-Content -Encoding ascii -LiteralPath $definition
    $output = Join-Path $objects "$libraryName.lib"
    & lib.exe /nologo /machine:x64 "/def:$definition" "/out:$output" | Out-Host
    if ($LASTEXITCODE) { throw "Import library generation failed for $libraryName." }
    return $output
}

Push-Location
try {
    & $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
    Set-Location -LiteralPath $objects
    $base = New-ImportLibrary 'ggml-base' 'ggml_[A-Za-z0-9_]+'
    $registry = New-ImportLibrary 'ggml_llamacpp' 'ggml_[A-Za-z0-9_]+'
    $server = New-ImportLibrary 'llama-server-impl' '\?llama_server@@YAHHPEAPEAD@Z'
    $flags = @('/nologo', '/MD', '/O2', '/EHsc', '/std:c++17', '/Zc:__cplusplus', '/utf-8', '/DNDEBUG', '/DGGML_SHARED', "/I$inc")
    & cl.exe @flags (Join-Path $src 'launcher.cpp') /link $base $registry $server "/out:$build\phoenix-server.exe"
    if ($LASTEXITCODE) { throw 'Building the Phoenix server launcher failed.' }

    if (-not $LauncherOnly) {
        $cpu = New-ImportLibrary 'ggml-cpu' 'ggml_[A-Za-z0-9_]+'
        $backendFlags = @('/LD', '/DGGML_BACKEND_SHARED', '/DGGML_BACKEND_DL', '/DGGML_BACKEND_BUILD', "/I$internals", "/I$XrtSdk\include")
        & cl.exe @flags @backendFlags (Join-Path $src 'ggml-phoenix.cpp') (Join-Path $src 'phoenix_matvec.cpp') /link $base $cpu $registry (Join-Path $XrtSdk 'lib\xrt_coreutil.lib') "/implib:$objects\ggml-phoenix.lib" "/out:$build\ggml-phoenix.dll"
        if ($LASTEXITCODE) { throw 'Building the Phoenix GGML backend failed.' }
    }
} finally {
    Pop-Location
}
