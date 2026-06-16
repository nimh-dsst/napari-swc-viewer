$ErrorActionPreference = "Stop"

$repoDir = Join-Path $env:USERPROFILE "repos\napari-swc-viewer"

if (-not (Test-Path -LiteralPath $repoDir)) {
    Write-Host "Could not find the repository at:"
    Write-Host "  $repoDir"
    Write-Host ""
    Write-Host "Clone the repository to %USERPROFILE%\repos\napari-swc-viewer, or edit this script to use your actual path."
    Write-Host ""
    Read-Host "Press Enter to close this window"
    exit 1
}

Set-Location -LiteralPath $repoDir

try {
    & pixi run napari
    $status = $LASTEXITCODE
} catch {
    Write-Host ""
    Write-Host "Failed to run 'pixi run napari'."
    Write-Host $_.Exception.Message
    Write-Host ""
    Read-Host "Press Enter to close this window"
    exit 1
}

Write-Host ""
if ($status -eq 0) {
    Write-Host "napari has closed."
} else {
    Write-Host "pixi run napari exited with status $status."
}

Read-Host "Press Enter to close this window"
exit $status
