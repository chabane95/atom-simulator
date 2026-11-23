param(
    [ValidateSet('view', 'image', 'render')]
    [string]$Mode = 'view'
)

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $projectPython)) {
        Write-Host 'Python-Umgebung wird eingerichtet ...'
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 oder neuer wird benoetigt.' }
        & $projectPython -m pip install -e .
        if ($LASTEXITCODE -ne 0) { throw 'Die Abhaengigkeiten konnten nicht installiert werden.' }
    }
    & $projectPython -m atom_simulator $Mode
    if ($LASTEXITCODE -ne 0) { throw 'atom-simulator wurde mit einem Fehler beendet.' }
}
finally {
    Pop-Location
}
