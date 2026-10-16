<#
.SYNOPSIS
    Script de control de infraestructura AWS para la PoC de MLOps (Churn Prediction).

.DESCRIPTION
    Permite pausar/reanudar la instancia EC2 o destruir/recrear toda la infraestructura
    con Terraform sin necesidad de actualizar los GitHub Secrets.

.EXAMPLE
    .\aws-control.ps1 status   # Ver estado actual e IP pública
    .\aws-control.ps1 stop     # Apagar solo la instancia EC2 (ahorra horas de cómputo, conserva Docker)
    .\aws-control.ps1 start    # Encender la instancia EC2 apagada (el contenedor levanta solo)
    .\aws-control.ps1 down     # Destruir TODA la infraestructura en AWS (terraform destroy)
    .\aws-control.ps1 up       # Crear/levantar TODA la infraestructura en AWS (terraform apply)
#>

param (
    [Parameter(Mandatory = $false, Position = 0)]
    [ValidateSet("status", "stop", "start", "down", "up")]
    [string]$Action = "status"
)

$ErrorActionPreference = "Stop"
$Region = "us-east-1"
$InstanceTagName = "mlops-churn-poc-api-server"
$TerraformDir = Join-Path $PSScriptRoot "terraform"

function Get-MlopsInstance {
    $json = aws ec2 describe-instances `
        --region $Region `
        --filters "Name=tag:Name,Values=$InstanceTagName" "Name=instance-state-name,Values=pending,running,stopping,stopped" `
        --query "Reservations[0].Instances[0].{InstanceId:InstanceId,State:State.Name,PublicIp:PublicIpAddress,Type:InstanceType}" `
        --output json 2>$null

    if (-not $json -or $json -eq "null") {
        return $null
    }
    return ($json | ConvertFrom-Json)
}

switch ($Action) {
    "status" {
        Write-Host "`n=== ESTADO DE INFRAESTRUCTURA AWS MLOPS ===" -ForegroundColor Cyan
        $inst = Get-MlopsInstance
        if (-not $inst -or -not $inst.InstanceId) {
            Write-Host "[!] No hay ninguna instancia EC2 activa o detenida ($InstanceTagName)." -ForegroundColor Yellow
            Write-Host "    Ejecuta '.\aws-control.ps1 up' para crear la infraestructura con Terraform."
        } else {
            Write-Host "Instancia ID : $($inst.InstanceId)"
            Write-Host "Tipo         : $($inst.Type)"
            Write-Host "Estado       : $($inst.State)" -ForegroundColor $(if ($inst.State -eq "running") { "Green" } else { "Yellow" })
            if ($inst.PublicIp) {
                Write-Host "IP Publica   : $($inst.PublicIp)"
                Write-Host "FastAPI Docs : http://$($inst.PublicIp):8000/docs" -ForegroundColor Green
                Write-Host "Health Check : http://$($inst.PublicIp):8000/health" -ForegroundColor Green
            }
        }
        Write-Host ""
    }

    "stop" {
        Write-Host "`n[*] Buscando instancia EC2 '$InstanceTagName' para detener..." -ForegroundColor Cyan
        $inst = Get-MlopsInstance
        if (-not $inst -or -not $inst.InstanceId) {
            Write-Host "[!] No se encontro la instancia EC2." -ForegroundColor Red
            exit 1
        }
        if ($inst.State -eq "stopped") {
            Write-Host "[OK] La instancia $($inst.InstanceId) ya esta detenida." -ForegroundColor Yellow
            exit 0
        }
        Write-Host "[*] Deteniendo instancia $($inst.InstanceId)..." -ForegroundColor Yellow
        aws ec2 stop-instances --region $Region --instance-ids $inst.InstanceId | Out-Null
        aws ec2 wait instance-stopped --region $Region --instance-ids $inst.InstanceId
        Write-Host "[OK] Instancia $($inst.InstanceId) detenida exitosamente (no consume horas EC2 y conserva el contenedor Docker)." -ForegroundColor Green
    }

    "start" {
        Write-Host "`n[*] Buscando instancia EC2 '$InstanceTagName' para iniciar..." -ForegroundColor Cyan
        $inst = Get-MlopsInstance
        if (-not $inst -or -not $inst.InstanceId) {
            Write-Host "[!] No existe la instancia EC2. Usa '.\aws-control.ps1 up' para crearla con Terraform." -ForegroundColor Red
            exit 1
        }
        if ($inst.State -eq "running") {
            Write-Host "[OK] La instancia $($inst.InstanceId) ya esta en ejecucion en http://$($inst.PublicIp):8000/docs" -ForegroundColor Green
            exit 0
        }
        Write-Host "[*] Iniciando instancia $($inst.InstanceId)..." -ForegroundColor Cyan
        aws ec2 start-instances --region $Region --instance-ids $inst.InstanceId | Out-Null
        Write-Host "[*] Esperando a que la instancia este en estado 'running'..." -ForegroundColor Cyan
        aws ec2 wait instance-running --region $Region --instance-ids $inst.InstanceId
        $inst = Get-MlopsInstance
        Write-Host "[OK] Instancia encendida. El contenedor Docker iniciara automaticamente (--restart always)." -ForegroundColor Green
        Write-Host "     FastAPI Docs : http://$($inst.PublicIp):8000/docs" -ForegroundColor Green
        Write-Host "     Health Check : http://$($inst.PublicIp):8000/health" -ForegroundColor Green
    }

    "down" {
        Write-Host "`n[!] Destruyendo TODA la infraestructura en AWS con Terraform..." -ForegroundColor Yellow
        Push-Location $TerraformDir
        try {
            terraform destroy -auto-approve
            Write-Host "`n[OK] Todos los recursos de AWS fueron eliminados (Costo $0)." -ForegroundColor Green
            Write-Host "     Cuando vuelvas a ejecutar '.\aws-control.ps1 up', NO necesitaras cambiar ningun Secret en GitHub." -ForegroundColor Cyan
        } finally {
            Pop-Location
        }
    }

    "up" {
        Write-Host "`n[*] Aprovisionando infraestructura en AWS con Terraform..." -ForegroundColor Cyan
        Push-Location $TerraformDir
        try {
            terraform init -upgrade=false
            terraform apply -auto-approve
            Write-Host "`n[OK] Infraestructura lista en AWS." -ForegroundColor Green
            Write-Host "     NOTA: Si acabas de recrear desde cero ('down' -> 'up'), ve a la pestaña 'Actions' en GitHub" -ForegroundColor Yellow
            Write-Host "     y ejecuta el workflow (o haz un git push a main) para subir la imagen Docker a ECR y lanzarla en la nueva EC2." -ForegroundColor Yellow
        } finally {
            Pop-Location
        }
    }
}
