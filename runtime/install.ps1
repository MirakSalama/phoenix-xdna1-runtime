[CmdletBinding()]param(
    [string]$StageOnly,
    [switch]$CheckOnly
)
$ErrorActionPreference='Stop'
$lock=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'runtime-lock.json') -Raw | ConvertFrom-Json -AsHashtable
$backends=(Resolve-Path -LiteralPath (Join-Path $env:USERPROFILE '.lmstudio/extensions/backends')).Path
$base=Join-Path $backends $lock.vulkan
$npu=Join-Path $backends $lock.phoenix
$target=if ($StageOnly) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($StageOnly) } else { Join-Path $backends "$($lock.name)-$($lock.version)" }
$launcher=Join-Path $PSScriptRoot 'build/phoenix-hybrid.exe'
if (!(Test-Path -LiteralPath $launcher -PathType Leaf)) {throw 'Build the launcher or use the prebuilt experimental package.'}
$localFiles=@{
    'llama-server.exe' = $launcher
    'ggml-phoenix.dll' = Join-Path $PSScriptRoot 'build/ggml-phoenix.dll'
    'programs/bonsai-q1-3072x5120.xclbin' = Join-Path $PSScriptRoot 'programs/bonsai-q1-3072x5120.xclbin'
    'programs/bonsai-q1-3072x5120.bin' = Join-Path $PSScriptRoot 'programs/bonsai-q1-3072x5120.bin'
}
if ($null -eq $lock.payloadFiles -or $lock.payloadFiles.Count -ne $localFiles.Count) {throw 'The candidate payload checksum lock is incomplete.'}
foreach ($name in $localFiles.Keys) {
    if (!(Test-Path -LiteralPath $localFiles[$name] -PathType Leaf) -or
        (Get-FileHash -LiteralPath $localFiles[$name]).Hash -ne $lock.payloadFiles[$name]) {throw "Candidate payload missing or changed: $name"}
}

function Assert-Inputs($root,$files) {
    foreach ($name in $files.Keys) {
        if ($name -notmatch '^(programs/)?[a-zA-Z0-9_.-]+$' -or $name.Contains('..')) {throw 'Unsafe input lock path'}
        $src=Join-Path $root $name
        if (!(Test-Path -LiteralPath $src -PathType Leaf) -or (Get-FileHash -LiteralPath $src).Hash -ne $files[$name]) {
            throw "Pinned runtime file missing or changed: $name"
        }
    }
}
$retained=@{}
foreach ($name in $lock.vulkanFiles.Keys) {
    if ($name -notin 'backend-manifest.json','display-data.json','engine-protocol-server-artifacts.json','llama-server.exe') {$retained[$name]=$lock.vulkanFiles[$name]}
}
$class='HKLM:/SYSTEM/CurrentControlSet/Control/Class/{4d36e968-e325-11ce-bfc1-08002be10318}'
$drivers=@(Get-ChildItem $class -ErrorAction SilentlyContinue | ForEach-Object {$_.GetValue('VulkanDriverName')} |
    Where-Object {$_ -like '*\amd-vulkan64.json' -and (Test-Path -LiteralPath $_ -PathType Leaf)} | Sort-Object -Unique)
$driverPath = if ($drivers.Count -eq 1) {[string]$drivers[0]} else {''}
if (Test-Path -LiteralPath $target) {
    if ((Get-Item -Force -LiteralPath $target).Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Unexpected linked runtime directory.'}
    $manifest=Get-Content -LiteralPath (Join-Path $target 'backend-manifest.json') -Raw | ConvertFrom-Json
    if ($manifest.name -ne $lock.name -or $manifest.version -ne $lock.version -or
        (Get-FileHash -LiteralPath "$target/llama-server.exe").Hash -ne (Get-FileHash -LiteralPath $launcher).Hash) {
        throw "Another runtime already occupies $target"
    }
    Assert-Inputs $target $retained
    Assert-Inputs $target $lock.phoenixFiles
    Assert-Inputs $target $lock.payloadFiles
    $icdPath=Join-Path $target 'amd-icd.txt'
    $currentDriver=if (Test-Path -LiteralPath $icdPath -PathType Leaf) {(Get-Content -Raw -LiteralPath $icdPath).Trim()}
    if ($currentDriver -ne $driverPath) {
        if ($CheckOnly) {throw 'The installed AMD driver path has changed. Re-run install.ps1 to refresh it.'}
        $tempDriver=Join-Path $target ('.driver-'+[guid]::NewGuid().ToString('N'))
        try {
            [IO.File]::WriteAllText($tempDriver,$driverPath+[Environment]::NewLine,[Text.Encoding]::ASCII)
            [IO.File]::Move($tempDriver,$icdPath,$true)
        } finally {
            if (Test-Path -LiteralPath $tempDriver) {Remove-Item -LiteralPath $tempDriver}
        }
        Write-Output 'Updated the AMD Vulkan driver path for subsequent workers.'
    }
    Write-Output "Already installed at $target"
    return
}
foreach ($source in $base,$npu) {
    if (!(Test-Path -LiteralPath $source -PathType Container)) {throw "Install the exact pinned source runtime first: $source"}
}
Assert-Inputs $base $lock.vulkanFiles
Assert-Inputs $npu $lock.phoenixFiles
if ($CheckOnly) {Write-Output "Verified $($lock.vulkan), $($lock.phoenix) and the candidate payload. Radeon driver available: $([bool]$driverPath)"; return}
$parent=[IO.Path]::GetDirectoryName($target)
$null=[IO.Directory]::CreateDirectory($parent)
$stagingParent=if ($StageOnly) {$parent} else {[IO.Path]::GetDirectoryName($backends)}
$tmp=Join-Path $stagingParent ('.hybrid-stage-'+[guid]::NewGuid().ToString('N'))
$null=New-Item -ItemType Directory -Path $tmp
try {
    foreach ($name in $lock.vulkanFiles.Keys) {
        if ($name -ne 'llama-server.exe') { Copy-Item -LiteralPath (Join-Path $base $name) -Destination (Join-Path $tmp $name) }
    }
    $null=[IO.Directory]::CreateDirectory((Join-Path $tmp 'programs'))
    foreach ($name in $lock.phoenixFiles.Keys) {
        Copy-Item -LiteralPath (Join-Path $npu $name) -Destination (Join-Path $tmp $name)
    }
    foreach ($name in $localFiles.Keys) {
        Copy-Item -LiteralPath $localFiles[$name] -Destination (Join-Path $tmp $name)
    }
    [IO.File]::WriteAllText("$tmp/amd-icd.txt",$driverPath+[Environment]::NewLine,[Text.Encoding]::ASCII)
    $manifest=Get-Content -Raw -LiteralPath "$tmp/backend-manifest.json" | ConvertFrom-Json
    $manifest.name=$lock.name
    $manifest.version=$lock.version
    $manifest.domains=@('llm')
    $manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath "$tmp/backend-manifest.json" -Encoding utf8NoBOM
    $label = if ($lock.name -eq 'phoenix-bonsai-preview-win-x86_64-avx2') {'Phoenix Bonsai XDNA 1 (Release Candidate)'} else {'Phoenix XDNA 1 (Windows)'}
    $display = [object[]]::new(1)
    $display[0] = @('en',@{langKey='en';displayName=$label;description='GGUF inference with CPU and XDNA 1, optional Radeon placements, and packed one-bit Bonsai output scoring.';releaseNotes=@()})
    ConvertTo-Json -Depth 6 -InputObject $display | Set-Content -LiteralPath "$tmp/display-data.json" -Encoding utf8NoBOM
    $art=Get-Content -Raw -LiteralPath "$tmp/engine-protocol-server-artifacts.json" | ConvertFrom-Json
    foreach ($name in @($lock.phoenixFiles.Keys)+@($lock.payloadFiles.Keys | Where-Object {$_ -ne 'llama-server.exe'})+@('amd-icd.txt')) {
        $art.files+= [pscustomobject]@{relative_path=$name;executable=$false}
    }
    $art | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath "$tmp/engine-protocol-server-artifacts.json" -Encoding utf8NoBOM
    Assert-Inputs $tmp $retained
    Assert-Inputs $tmp $lock.phoenixFiles
    Assert-Inputs $tmp $lock.payloadFiles
    [IO.Directory]::Move($tmp,$target)
} finally {
    if (Test-Path -LiteralPath $tmp) {
        $resolved=(Resolve-Path -LiteralPath $tmp).Path
        if ([IO.Path]::GetDirectoryName($resolved) -ne $stagingParent -or [IO.Path]::GetFileName($resolved) -notmatch '^\.hybrid-stage-[0-9a-f]{32}$' -or
            ((Get-Item -Force -LiteralPath $resolved).Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            (Get-ChildItem -LiteralPath $resolved -Recurse -Force | Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint})) {throw 'Unexpected temporary installation path'}
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}
Write-Output "Prepared $($lock.name)@$($lock.version) at $target"
