# Install a separate Bionic pack using the bindings from the matching local CPU runtime.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$extensions = (Resolve-Path -LiteralPath (Join-Path $env:USERPROFILE '.lmstudio\extensions')).Path
$backends = (Resolve-Path -LiteralPath (Join-Path $extensions 'backends')).Path
$name = 'phoenix-xdna1-win-x86_64-avx2'
$version = '0.1.0'
$official = Join-Path $backends 'llama.cpp-win-x86_64-avx2-2.80.1'
$target = Join-Path $backends "$name-$version"
$binaries = @{
    'llama-server.exe' = 'build\phoenix-server.exe'
    'ggml-phoenix.dll' = 'build\ggml-phoenix.dll'
    'programs\gemv-3072x1024.xclbin' = 'programs\gemv-3072x1024.xclbin'
    'programs\gemv-3072x1024.bin' = 'programs\gemv-3072x1024.bin'
}
foreach ($relative in $binaries.Values) {
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $relative) -PathType Leaf)) { throw "Missing build output $relative" }
}
if (-not (Test-Path -LiteralPath (Join-Path $env:SystemRoot 'System32\xrt_coreutil.dll') -PathType Leaf)) {
    throw 'The AMD XRT runtime for Windows is required.'
}
$pinned = @{
    'ggml-base.dll' = 'B3C7555601B62248F22413135CBC29B81613176D32DC2CA4C4DBBA6D4E36F410'
    'llama-server-impl.dll' = 'E12AAA278E59F20CE92E80BA07B2BBE6147E7BB9FA9BDBFF969BFCFB6E349551'
}
foreach ($file in $pinned.Keys) {
    if ((Get-FileHash -LiteralPath (Join-Path $official $file)).Hash -ne $pinned[$file]) {
        throw "The installed CPU llama.cpp 2.80.1 differs from the build used for this pack: $file"
    }
}
if (Test-Path -LiteralPath $target) {
    foreach ($file in $binaries.Keys) {
        if ((Get-FileHash -LiteralPath (Join-Path $target $file)).Hash -ne
            (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $binaries[$file])).Hash) {
            throw 'A different Phoenix pack is installed at the same version. Unload it and remove that pack in Bionic before reinstalling.'
        }
    }
    Write-Output "Already installed: $name@$version"
    return
}

$stageName = '.phoenix-xdna1-stage-' + [guid]::NewGuid().ToString('N')
$stage = Join-Path $extensions $stageName
$null = New-Item -ItemType Directory -Path $stage
try {
    # Native bindings stay with their original version; they run model inspection.
    foreach ($f in Get-ChildItem -LiteralPath $official -File) {
        if (-not $binaries.ContainsKey($f.Name)) { Copy-Item -LiteralPath $f.FullName -Destination $stage }
    }
    $null = New-Item -ItemType Directory -Path (Join-Path $stage 'programs')
    foreach ($file in $binaries.Keys) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $binaries[$file]) -Destination (Join-Path $stage $file)
    }
    $manifestPath = Join-Path $stage 'backend-manifest.json'
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    $manifest.name = $name
    $manifest.version = $version
    $manifest.domains = @('llm')
    $manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding utf8NoBOM
    '[["en",{"langKey":"en","displayName":"Phoenix XDNA 1 (Windows)","description":"Runs supported GGUF decode matrix-vector operations on the AMD XDNA 1 NPU, with CPU execution for other operations.","releaseNotes":[]}]]' |
        Set-Content -LiteralPath (Join-Path $stage 'display-data.json') -Encoding utf8NoBOM
    $artPath = Join-Path $stage 'engine-protocol-server-artifacts.json'
    $artifacts = Get-Content -Raw -LiteralPath $artPath | ConvertFrom-Json
    foreach ($file in $binaries.Keys | Sort-Object) {
        $relative = $file.Replace('\','/')
        if ($artifacts.files.relative_path -notcontains $relative) {
            $artifacts.files += [pscustomobject]@{ relative_path = $relative; executable = $false }
        }
    }
    $artifacts | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $artPath -Encoding utf8NoBOM
    if ((Resolve-Path -LiteralPath $stage).Path -ne (Join-Path $extensions $stageName) -or
        [IO.Path]::GetFullPath($target) -ne (Join-Path $backends "$name-$version") -or (Test-Path -LiteralPath $target)) {
        throw 'Installation paths changed during staging.'
    }
    [IO.Directory]::Move($stage, $target)
} finally {
    if ((Test-Path -LiteralPath $stage) -and (Resolve-Path -LiteralPath $stage).Path -eq (Join-Path $extensions $stageName) -and
        -not ((Get-Item -LiteralPath $stage).Attributes -band [IO.FileAttributes]::ReparsePoint) -and
        -not (Get-ChildItem -LiteralPath $stage -Force -Recurse | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
}
Write-Output "Installed: $name@$version"
