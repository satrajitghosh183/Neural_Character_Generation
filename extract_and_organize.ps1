# Set your project root
$root = "D:\Master Things\Spring Sem Classes\PrincetonNeuralRendering\Project\Project_Programming"

# Where you want to store the organized dataset
$datasetRoot = Join-Path $root "dataset"

# Create dataset root if it doesn't exist
if (!(Test-Path -Path $datasetRoot)) {
    New-Item -ItemType Directory -Path $datasetRoot | Out-Null
}

# Find all .tar files recursively
Get-ChildItem -Path $root -Recurse -Filter *.tar | ForEach-Object {

    $tarPath = $_.FullName
    $fileName = $_.Name

    Write-Host ""
    Write-Host "Found archive: $fileName"

    # Extract entity ID (e.g., 6795937)
    if ($fileName -match "^(\d{7})") {
        $entityId = $matches[1]
    } else {
        Write-Host "WARNING: Could not detect entity ID. Skipping: $fileName"
        return
    }

    # Determine asset type
    if ($fileName -match "images--") {
        $type = "images"
    } elseif ($fileName -match "tracked_mesh--") {
        $type = "tracked_mesh"
    } elseif ($fileName -match "metadata") {
        $type = "metadata"
    } elseif ($fileName -match "texture" -or $fileName -match "unwrapped_uv") {
        $type = "texture"
    } elseif ($fileName -match "audio") {
        $type = "audio"
    } else {
        $type = "other"
    }

    # Destination: dataset/ENTITY_ID/ASSET_TYPE/
    $destDir = Join-Path -Path $datasetRoot -ChildPath "$entityId\$type"

    # Make destination folder
    if (!(Test-Path -Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }

    Write-Host "Extracting to: $destDir"

    # Extract the .tar file
    tar -xf $tarPath -C $destDir

    Write-Host "Done extracting: $fileName"
}
