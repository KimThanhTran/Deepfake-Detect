# Requires: Python with pip, internet connection, and tar (Windows 10+ includes tar)
# This script downloads datasets used by NPR-DeepfakeDetection to the local "dataset" folder.
# It mirrors download_dataset.sh functionality for Windows PowerShell.

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[ERR ] $msg"  -ForegroundColor Red }

# Resolve repo root based on script location
$ScriptDir = Split-Path -Parent $PSCommandPath
Set-Location $ScriptDir
Write-Info "Repo root: $ScriptDir"

$datasetRoot = Join-Path $ScriptDir 'dataset'
New-Item -ItemType Directory -Force -Path $datasetRoot | Out-Null
Write-Info "Download root: $datasetRoot"

function Get-PythonCmd {
    # Prefer 'python' but fall back to 'py'
    if (Get-Command python -ErrorAction SilentlyContinue) { return 'python' }
    if (Get-Command py -ErrorAction SilentlyContinue)      { return 'py' }
    throw "Python is not installed or not on PATH. Please install Python 3.x."
}

function Ensure-GDown {
    $py = Get-PythonCmd
    # Check if gdown is importable
    & $py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('gdown') else 1)"
    if ($LASTEXITCODE -eq 0) {
        Write-Info "gdown is available."
        return
    }
    Write-Info "Installing gdown==4.7.1 ..."
    & $py -m pip install "gdown==4.7.1"
    Write-Info "gdown installed."
}

function Invoke-GDownFolder {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$OutDir
    )
    $py = Get-PythonCmd
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Write-Info "Downloading Google Drive folder: $Url -> $OutDir"
    & $py -c "import sys, gdown; sys.exit(0 if gdown.download_folder(url=r'$Url', output=r'$OutDir', use_cookies=False, quiet=False) else 1)"
}

function Invoke-GDownFile {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$OutFile
    )
    $py = Get-PythonCmd
    Write-Info "Downloading file: $Url -> $OutFile"
    & $py -c "import sys, gdown; sys.exit(0 if gdown.download(url=r'$Url', output=r'$OutFile', use_cookies=False, quiet=False) else 1)"
}

function Expand-TarGzInFolder {
    param([Parameter(Mandatory=$true)][string]$Folder)

    $archives = Get-ChildItem -Path $Folder -File -Include *.tar.gz, *.tgz
    foreach ($a in $archives) {
        Write-Info "Extracting: $($a.FullName)"
        # Ensure we extract into the folder containing the archive
        tar -xf $a.FullName -C $a.DirectoryName
        Remove-Item $a.FullName -Force
    }
}

function Expand-ZipsInFolder {
    param([Parameter(Mandatory=$true)][string]$Folder)

    $zips = Get-ChildItem -Path $Folder -File -Include *.zip
    foreach ($z in $zips) {
        Write-Info "Extracting ZIP: $($z.FullName)"
        $dest = $z.DirectoryName
        try {
            Expand-Archive -Path $z.FullName -DestinationPath $dest -Force
            Remove-Item $z.FullName -Force
        } catch {
            # Fallback to tar if Expand-Archive fails
            Write-Warn "Expand-Archive failed, trying tar for: $($z.Name)"
            tar -xf $z.FullName -C $dest
            Remove-Item $z.FullName -Force
        }
    }
}

function Expand-ArchivesRecursive {
    param([Parameter(Mandatory=$true)][string]$Root)

    Write-Info "Recursively extracting archives under: $Root"
    # First expand zips then tarballs (order not critical but consistent)
    $zips = Get-ChildItem -Path $Root -Recurse -File -Include *.zip
    foreach ($z in $zips) {
        Write-Info "Extracting ZIP: $($z.FullName)"
        $dest = $z.DirectoryName
        try {
            Expand-Archive -Path $z.FullName -DestinationPath $dest -Force
            Remove-Item $z.FullName -Force
        } catch {
            Write-Warn "Expand-Archive failed, trying tar for: $($z.Name)"
            tar -xf $z.FullName -C $dest
            Remove-Item $z.FullName -Force
        }
    }
    $tars = Get-ChildItem -Path $Root -Recurse -File -Include *.tar.gz, *.tgz
    foreach ($t in $tars) {
        Write-Info "Extracting TAR.GZ: $($t.FullName)"
        tar -xf $t.FullName -C $t.DirectoryName
        Remove-Item $t.FullName -Force
    }
}

function Merge-And-Extract-SplitTarGz {
    param([Parameter(Mandatory=$true)][string]$Folder)

    # Find all parts like name.tar.gz.aa, name.tar.gz.ab ... and group by base 'name.tar.gz'
    $all = Get-ChildItem -Path $Folder -File -Filter "*.tar.gz.*" -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch '\.tar\.gz$' }
    if (-not $all) { return }

    $groups = $all | Group-Object -Property { $_.Name -replace '\.tar\.gz\.[^.]+$','' }
    foreach ($g in $groups) {
        $baseName = $g.Name  # like 'train' or 'test'
        $merged = Join-Path $Folder ("{0}.tar.gz" -f $baseName)
        if (Test-Path $merged) { Remove-Item $merged -Force }
        Write-Info ("Merging {0} parts -> {1}" -f $baseName, $merged)
        $fs = [System.IO.File]::OpenWrite($merged)
        try {
            foreach ($p in ($g.Group | Sort-Object Name)) {
                Write-Info "Appending part $($p.Name)"
                $in = [System.IO.File]::OpenRead($p.FullName)
                try {
                    $buffer = New-Object byte[] 1048576
                    while (($read = $in.Read($buffer,0,$buffer.Length)) -gt 0) {
                        $fs.Write($buffer,0,$read)
                    }
                } finally { $in.Close() }
            }
        } finally { $fs.Close() }

        Write-Info "Extracting $merged"
        tar -xf $merged -C $Folder
        Remove-Item $merged -Force
        foreach ($p in $g.Group) { Remove-Item $p.FullName -Force }
    }
}

# 1) Ensure gdown is present
Ensure-GDown

# 2) Download Google Drive folders
$jobs = @(
    @{ Url = 'https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-'; Name = 'UniversalFakeDetect' },
    @{ Url = 'https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj'; Name = 'GANGen-Detection' },
    @{ Url = 'https://drive.google.com/drive/folders/1tKsOU-6FDdstrrKLPYuZ7RpQwtOSHxUD'; Name = 'DiffusionForensics' },
    @{ Url = 'https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq'; Name = 'Diffusion1kStep' }
)

foreach ($job in $jobs) {
    $out = Join-Path $datasetRoot $job.Name
    Invoke-GDownFolder -Url $job.Url -OutDir $out
    # Unpack archives within each dataset folder (recursively to handle gdown-created subfolders)
    Expand-ArchivesRecursive -Root $out
}

# 3) Optional: AIGCDetect testset (requires merged multi-part zip). Attempt only if 7z is available.
try {
    $aigcFolder = Join-Path $datasetRoot 'AIGCDetect_testset'
    Invoke-GDownFolder -Url 'https://drive.google.com/drive/folders/1BUv1MT1cm90QN3WTMHLEr8PXBsKGxKC9' -OutDir $aigcFolder
    $has7z = $null -ne (Get-Command 7z -ErrorAction SilentlyContinue)
    if ($has7z) {
        Write-Info 'Merging and extracting multi-part AIGCDetect zip with 7z.'
        Push-Location $aigcFolder
        # 7z automatically handles split archives when extracting from the .zip part
        if (Test-Path 'test.zip') {
            7z x 'test.zip' -y -o.
        } else {
            Write-Warn 'AIGCDetect split zip entry test.zip not found; skipping extraction.'
        }
        Pop-Location
    } else {
        Write-Warn '7-Zip not found. If AIGCDetect contains split zips (test.z01, test.zip), install 7-Zip and extract test.zip.'
    }
} catch {
    Write-Warn "AIGCDetect_testset download/extraction encountered an issue: $($_.Exception.Message)"
}

# 4) CNNDetection ForenSynths testset
$cnnZip = Join-Path $datasetRoot 'CNN_synth_testset.zip'
Invoke-GDownFile -Url 'https://drive.google.com/u/0/uc?id=1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1' -OutFile $cnnZip
$forenOut = Join-Path $datasetRoot 'ForenSynths'
New-Item -ItemType Directory -Force -Path $forenOut | Out-Null
try {
    Expand-Archive -Path $cnnZip -DestinationPath $forenOut -Force
} catch {
    Write-Warn 'Expand-Archive failed for CNN_synth_testset.zip, trying tar.'
    tar -xf $cnnZip -C $forenOut
}
if (Test-Path $cnnZip) { Remove-Item $cnnZip -Force }

# 5) If DiffusionForensics missing (Table3), attempt download again
$diffForensics = Join-Path $datasetRoot 'DiffusionForensics'
if (-not (Test-Path $diffForensics)) {
    try {
        Invoke-GDownFolder -Url 'https://drive.google.com/drive/folders/1tKsOU-6FDdstrrKLPYuZ7RpQwtOSHxUD' -OutDir $diffForensics
        Expand-ArchivesRecursive -Root $diffForensics
    } catch {
        Write-Warn "DiffusionForensics download encountered an issue: $($_.Exception.Message)"
    }
}

# 6) Post-process Diffusion1kStep to flatten expected folders to top-level
$d1k = Join-Path $datasetRoot 'Diffusion1kStep'
if (Test-Path $d1k) {
    Expand-ArchivesRecursive -Root $d1k
    $expected = @('DALLE','ddpm','guided-diffusion','improved-diffusion','midjourney')
    foreach ($name in $expected) {
        $candidates = Get-ChildItem -Path $d1k -Recurse -Directory -Filter $name | Sort-Object FullName | Select-Object -First 1
        if ($candidates) {
            $src = $candidates.FullName
            $dst = Join-Path $d1k $name
            if ($src -ne $dst -and -not (Test-Path $dst)) {
                Write-Info "Flattening $name to $dst"
                Move-Item -Force -Path $src -Destination $dst
            }
        }
    }
    # Remove timestamped or drive-download helper folders now that contents are flattened
    $toRemove = Get-ChildItem -Path $d1k -Directory | Where-Object { $_.Name -like 'Diffusion1kStep-*' -or $_.Name -like 'drive-download-*' }
    foreach ($dir in $toRemove) {
        Write-Info "Removing temp folder: $($dir.FullName)"
        Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# 7) Merge and extract ForenSynths multi-part archives; flatten nested val/train if double nested
$forenSynths = Join-Path $datasetRoot 'ForenSynths'
if (Test-Path $forenSynths) {
    Merge-And-Extract-SplitTarGz -Folder $forenSynths
    # Flatten val/val and train/train if exists
    $valNested = Join-Path $forenSynths 'val/val'
    if (Test-Path $valNested) {
        Write-Info "Flattening nested val folder"
        Get-ChildItem -Path $valNested -Directory | ForEach-Object { Move-Item -Force -Path $_.FullName -Destination (Join-Path $forenSynths 'val') }
        Remove-Item -Recurse -Force $valNested -ErrorAction SilentlyContinue
    }
    $trainNested = Join-Path $forenSynths 'train/train'
    if (Test-Path $trainNested) {
        Write-Info "Flattening nested train folder"
        Get-ChildItem -Path $trainNested -Directory | ForEach-Object { Move-Item -Force -Path $_.FullName -Destination (Join-Path $forenSynths 'train') }
        Remove-Item -Recurse -Force $trainNested -ErrorAction SilentlyContinue
    }
    # Remove helper script if present
    $decompress = Join-Path $forenSynths 'decompress.sh'
    if (Test-Path $decompress) { Remove-Item $decompress -Force }
}

# 8) Extract DiffusionForensics timestamped archives and flatten expected structure
$diffF = Join-Path $datasetRoot 'DiffusionForensics'
if (Test-Path $diffF) {
    # Extract any tar.gz inside timestamp folders
    $tarFiles = Get-ChildItem -Path $diffF -Recurse -File -Include *.tar.gz, *.tgz
    foreach ($t in $tarFiles) {
        Write-Info "Extracting DiffusionForensics archive: $($t.FullName)"
        tar -xf $t.FullName -C $t.DirectoryName
        Remove-Item $t.FullName -Force
    }
    # Expected subfolders after extraction (may appear inside nested folders): adm, ddpm, iddpm, ldm, pndm, sdv1_new, sdv2, vqdiffusion, stylegan_official, projectedgan, diff-projectedgan, diff-stylegan, if, midjourney, dalle2
    $expectedDF = @('adm','ddpm','iddpm','ldm','pndm','sdv1_new','sdv2','vqdiffusion','stylegan_official','projectedgan','diff-projectedgan','diff-stylegan','if','midjourney','dalle2')
    foreach ($name in $expectedDF) {
        $candidate = Get-ChildItem -Path $diffF -Recurse -Directory -Filter $name | Sort-Object FullName | Select-Object -First 1
        if ($candidate) {
            $dst = Join-Path $diffF $name
            if ($candidate.FullName -ne $dst -and -not (Test-Path $dst)) {
                Write-Info "Flattening DiffusionForensics $name"
                Move-Item -Force -Path $candidate.FullName -Destination $dst
            }
        }
    }
    # Remove timestamp folders now
    Get-ChildItem -Path $diffF -Directory | Where-Object { $_.Name -match 'googledrive-' -or $_.Name -match '-\d{8}T' } | ForEach-Object {
        Write-Info "Removing temp DF folder: $($_.FullName)"; Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue }
}

Write-Host "`nAll queued downloads/extractions complete. Note: Train/Val set (ForenSynths_train_val) is hosted on Baidudrive and must be downloaded manually: https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b" -ForegroundColor Green
