# Script to prepare files for Kaggle upload
# Run this before uploading to Kaggle

Write-Host "Preparing files for Kaggle upload..." -ForegroundColor Green

# Create output directory
$outputDir = "kaggle_upload"
if (Test-Path $outputDir) {
    Remove-Item -Recurse -Force $outputDir
}
New-Item -ItemType Directory -Path $outputDir | Out-Null
New-Item -ItemType Directory -Path "$outputDir/networks" | Out-Null
New-Item -ItemType Directory -Path "$outputDir/data" | Out-Null
New-Item -ItemType Directory -Path "$outputDir/options" | Out-Null

# Copy essential files
Write-Host "`nCopying code files..." -ForegroundColor Yellow

# Main scripts
Copy-Item "train_frequency.py" -Destination $outputDir
Copy-Item "evaluate_hybrid.py" -Destination $outputDir
Copy-Item "visualize_frequency.py" -Destination $outputDir
Copy-Item "util.py" -Destination $outputDir

# Networks
Copy-Item "networks/__init__.py" -Destination "$outputDir/networks/"
Copy-Item "networks/base_model.py" -Destination "$outputDir/networks/"
Copy-Item "networks/resnet.py" -Destination "$outputDir/networks/"
Copy-Item "networks/frequency_branch.py" -Destination "$outputDir/networks/"

# Data
Copy-Item "data/__init__.py" -Destination "$outputDir/data/"
Copy-Item "data/datasets.py" -Destination "$outputDir/data/"

# Options
Copy-Item "options/__init__.py" -Destination "$outputDir/options/"
Copy-Item "options/base_options.py" -Destination "$outputDir/options/"
Copy-Item "options/train_options.py" -Destination "$outputDir/options/"

# Documentation
Copy-Item "FREQUENCY_BRANCH_GUIDE.md" -Destination $outputDir
Copy-Item "KAGGLE_TRAINING_GUIDE.md" -Destination $outputDir

Write-Host "`n✅ Files copied to kaggle_upload/" -ForegroundColor Green

# Show structure
Write-Host "`nFolder structure:" -ForegroundColor Cyan
Get-ChildItem -Recurse $outputDir | Select-Object FullName | ForEach-Object {
    $_.FullName -replace [regex]::Escape($PWD.Path + "\$outputDir"), ""
}

# Calculate size
$totalSize = (Get-ChildItem -Recurse $outputDir | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "`nTotal size: $([math]::Round($totalSize, 2)) MB" -ForegroundColor Cyan

Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "NEXT STEPS:" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green
Write-Host "1. Zip folder 'kaggle_upload'" -ForegroundColor White
Write-Host "   Right-click → Send to → Compressed (zipped) folder" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Go to Kaggle: https://www.kaggle.com/code" -ForegroundColor White
Write-Host ""
Write-Host "3. Create New Notebook → Enable GPU T4/P100" -ForegroundColor White
Write-Host ""
Write-Host "4. Upload kaggle_upload.zip to notebook" -ForegroundColor White
Write-Host "   File → Upload Files" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Extract in notebook:" -ForegroundColor White
Write-Host "   !unzip kaggle_upload.zip -d . -q" -ForegroundColor Gray
Write-Host ""
Write-Host "6. Follow KAGGLE_TRAINING_GUIDE.md" -ForegroundColor White
Write-Host "="*60 -ForegroundColor Green
