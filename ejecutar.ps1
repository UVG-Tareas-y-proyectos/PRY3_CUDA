param([ValidateSet('run','test','benchmark')][string]$Accion='run')
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    & wsl -d Ubuntu -- bash scripts/run_wsl.sh $Accion
    $result=$LASTEXITCODE
} finally {
    Pop-Location
}
exit $result
