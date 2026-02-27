# Push current (clean) history to a new GitHub repository
# Run this AFTER creating the new repo on GitHub (do not add README/.gitignore/license)

param(
    [Parameter(Mandatory=$true)]
    [string]$NewRepoName
)

$repoUrl = "https://github.com/iamjamaal/$NewRepoName.git"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Push clean history to new repository" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "New repository URL: $repoUrl" -ForegroundColor Yellow
Write-Host ""
Write-Host "Make sure you have already created the repo on GitHub (empty, no README)." -ForegroundColor Yellow
Write-Host ""

# Remove neworigin if it already exists (e.g. from a previous run)
git remote remove neworigin 2>$null

Write-Host "Adding remote 'neworigin'..." -ForegroundColor Yellow
git remote add neworigin $repoUrl

Write-Host "Pushing main branch to new repository..." -ForegroundColor Yellow
git push neworigin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Done! Your clean history is now at:" -ForegroundColor Green
    Write-Host "  https://github.com/iamjamaal/$NewRepoName" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "The Contributors list on the new repo should show only you." -ForegroundColor Green
    Write-Host ""
    Write-Host "Optional - switch to new repo as default remote:" -ForegroundColor Yellow
    Write-Host "  git remote remove origin" -ForegroundColor White
    Write-Host "  git remote rename neworigin origin" -ForegroundColor White
    Write-Host "  git branch -u origin/main main" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Push failed. Check that:" -ForegroundColor Red
    Write-Host "  1. The repo '$NewRepoName' exists on GitHub" -ForegroundColor White
    Write-Host "  2. You have push access" -ForegroundColor White
    Write-Host "  3. The repo was created empty (no README)" -ForegroundColor White
}
