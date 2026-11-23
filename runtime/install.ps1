# Install a separate Bionic pack using the bindings from the matching local CPU runtime.
[CmdletBinding()]
param(
    [ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$compatibility = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'compatibility.json') | ConvertFrom-Json -AsHashtable
if (-not $Version) { $Version = $compatibility.version }
$extensions = (Resolve-Path -LiteralPath (Join-Path $env:USERPROFILE '.lmstudio\extensions')).Path
$backends = (Resolve-Path -LiteralPath (Join-Path $extensions 'backends')).Path
$name = $compatibility.pack
$official = Join-Path $backends "$($compatibility.basePack)-$($compatibility.baseVersion)"
$target = Join-Path $backends "$name-$version"
$binaries = @{
    'llama-server.exe' = 'build\phoenix-server.exe'
    'ggml-phoenix.dll' = 'build\ggml-phoenix.dll'
    'programs\stream-gemv.xclbin' = 'programs\stream-gemv.xclbin'
    'programs\prefill-2560x960x128.bin' = 'programs\prefill-2560x960x128.bin'
    'programs\prefill-2560x960x128.xclbin' = 'programs\prefill-2560x960x128.xclbin'
}
foreach ($rows in @(128,256,384,512,768,1024,1536,2048,2560,3072,4096,6144,8192,16384,32768,49152,65536)) {
    $relative = "programs\stream-$rows.bin"
    $binaries[$relative] = $relative
}
foreach ($relative in $binaries.Values) {
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $relative) -PathType Leaf)) { throw "Missing build output $relative" }
}
if (-not (Test-Path -LiteralPath (Join-Path $env:SystemRoot 'System32\xrt_coreutil.dll') -PathType Leaf)) {
    throw 'The AMD XRT runtime for Windows is required.'
}
$pinned = $compatibility.nativeHashes
foreach ($file in $pinned.Keys) {
    if ((Get-FileHash -LiteralPath (Join-Path $official $file)).Hash -ne $pinned[$file]) {
        throw "The installed CPU llama.cpp $($compatibility.baseVersion) differs from the build used for this pack: $file"
    }
}
if ($CheckOnly) {
    $versions = Get-ChildItem -Directory -LiteralPath $backends |
        Where-Object { $_.Name -match ('^' + [regex]::Escape($compatibility.basePack) + '-(\d+\.\d+\.\d+)$') } |
        ForEach-Object { [version]($_.Name.Substring($compatibility.basePack.Length + 1)) } | Sort-Object -Descending
    Write-Output "Verified CPU pack $($compatibility.baseVersion). Newest installed CPU pack: $($versions | Select-Object -First 1)."
    return
}
if (Test-Path -LiteralPath $target) {
    $existing = Get-Content -LiteralPath (Join-Path $target 'backend-manifest.json') -Raw | ConvertFrom-Json
    if ($existing.name -ne $name -or $existing.version -ne $Version -or
        $existing.engine_protocol_server.executable_relative_path -ne 'llama-server.exe') {
        throw 'The existing folder contains a different or incomplete runtime manifest.'
    }
    foreach ($file in $pinned.Keys) {
        if ((Get-FileHash -LiteralPath (Join-Path $target $file)).Hash -ne $pinned[$file]) {
            throw "The existing Phoenix pack contains an incompatible Bionic library: $file"
        }
    }
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
    & (Join-Path $stage 'llama-server.exe') --phoenix-self-test
    if ($LASTEXITCODE -ne 0) { throw "Staged Phoenix hardware self-test failed ($LASTEXITCODE)." }
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
