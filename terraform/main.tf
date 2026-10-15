# =========================================================================
# INFRAESTRUCTURA COMO CÓDIGO (IaC) - TERRAFORM PARA MLOPS EN AWS
# =========================================================================
# Este template provisiona:
# 1. Bucket S3 con versionado y cifrado para almacenamiento de artefactos de MLflow.
# 2. Security Group con reglas mínimas para HTTP y acceso a la API FastAPI.
# 3. Rol IAM e Instance Profile para acceso seguro a S3 y ECR sin hardcodear claves.
# 4. Instancia EC2 t2.micro (Free Tier) con Docker preinstalado vía User Data.
# =========================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # =========================================================================
  # PRODUCCIÓN: Backend Remoto con State Locking
  # =========================================================================
  # En producción, nunca almacenar terraform.tfstate localmente en el repositorio:
  # backend "s3" {
  #   bucket         = "mi-empresa-terraform-state"
  #   key            = "mlops/customer-churn/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "MLOps-Team"
    }
  }
}

# Sufijo aleatorio para garantizar unicidad global en el nombre del Bucket S3
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# =========================================================================
# 1. ALMACENAMIENTO DE ARTEFACTOS (MLflow Model Artifact Store)
# =========================================================================
# En producción:
# - MLflow Tracking Server apunta a este bucket mediante:
#   mlflow server --default-artifact-root s3://${aws_s3_bucket.mlflow_artifacts.id}/artifacts
# - Se configuraría un ciclo de vida (Lifecycle Rules) para transicionar modelos antiguos
#   a S3 Glacier Instant Retrieval / Deep Archive para optimizar costos de storage.

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket        = "${var.project_name}-mlflow-artifacts-${random_string.bucket_suffix.result}"
  force_destroy = true # En producción estricta: false para prevenir borrado accidental de modelos

  tags = {
    Name = "MLflow Model Artifacts Store"
  }
}

# Versionado obligatorio para gobernanza, linaje y rollback de modelos
resource "aws_s3_bucket_versioning" "mlflow_artifacts_versioning" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Cifrado en reposo (Server-Side Encryption)
resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts_crypto" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
      # En producción: utilizar AWS KMS gestionado por el cliente (CMK) para rotación de claves
    }
  }
}

# Bloqueo total de acceso público (Compliance & Seguridad)
resource "aws_s3_bucket_public_access_block" "mlflow_artifacts_public_block" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# =========================================================================
# 2. SEGURIDAD Y POLÍTICAS IAM (Principio de Menor Privilegio)
# =========================================================================
# La instancia EC2 asume un rol IAM temporal vía Instance Profile,
# evitando almacenar AWS_ACCESS_KEY_ID o AWS_SECRET_ACCESS_KEY en el servidor.

resource "aws_iam_role" "ec2_mlops_role" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# Política de lectura y descarga de modelos desde el bucket S3
resource "aws_iam_policy" "s3_read_models_policy" {
  name        = "${var.project_name}-s3-model-read-policy"
  description = "Permite a la API de inferencia descargar artefactos de modelos de MLflow"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_s3_policy" {
  role       = aws_iam_role.ec2_mlops_role.name
  policy_arn = aws_iam_policy.s3_read_models_policy.arn
}

# Permisos para AWS Systems Manager (SSM) para administración y despliegues sin abrir SSH
resource "aws_iam_role_policy_attachment" "attach_ssm_policy" {
  role       = aws_iam_role.ec2_mlops_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-instance-profile"
  role = aws_iam_role.ec2_mlops_role.name
}

# =========================================================================
# 3. RED Y SECURITY GROUPS
# =========================================================================
# Obtener la VPC por defecto para mantener la PoC simple y autocontenida
data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "api_sg" {
  name        = "${var.project_name}-api-sg"
  description = "Reglas de firewall para el microservicio de inferencia FastAPI"
  vpc_id      = data.aws_vpc.default.id

  # Puerto 80 para tráfico Web HTTP
  ingress {
    description = "Acceso HTTP publico"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Puerto 8000 directo para desarrollo / pruebas de FastAPI
  ingress {
    description = "Puerto directo FastAPI / Uvicorn"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # SSH (Puerto 22)
  ingress {
    description = "Acceso administrativo SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
  }

  # Salida libre para descargas de paquetes, imágenes Docker y comunicación con AWS
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-api-sg"
  }
}

# =========================================================================
# 4. INSTANCIA DE CÓMPUTO (EC2 para Hosting de Inferencia)
# =========================================================================
# Obtención dinámica de la AMI más reciente de Ubuntu 22.04 LTS
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "api_server" {
  ami                  = data.aws_ami.ubuntu.id
  instance_type        = var.instance_type
  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name
  vpc_security_group_ids = [aws_security_group.api_sg.id]

  # Provisionamiento automático con Docker y Docker Compose
  user_data = <<-EOF
              #!/bin/bash
              set -e
              apt-get update -y
              apt-get install -y apt-transport-https ca-certificates curl software-properties-common unzip
              curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
              echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
              apt-get update -y
              apt-get install -y docker-ce docker-ce-cli containerd.io
              systemctl enable docker
              systemctl start docker
              usermod -aG docker ubuntu

              # Instalar AWS CLI v2
              curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
              unzip -q awscliv2.zip
              ./aws/install

              echo "[MLOps Init] Servidor listo para hospedar contenedores de inferencia." > /var/log/mlops_init.log
              EOF

  root_block_device {
    volume_size           = 20
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = {
    Name = "${var.project_name}-api-server"
    Role = "Model-Serving-Host"
  }
}

# =========================================================================
# NOTAS DE ARQUITECTURA DE PRODUCCIÓN (ALTO VOLUMEN):
# =========================================================================
# En una arquitectura enterprise lista para producción:
# 1. Cómputo:
#    - Reemplazar EC2 standalone por Amazon ECS (Fargate) o Amazon EKS (Kubernetes)
#      con Horizontal Pod Autoscaler (HPA) basado en métricas de CPU o RPS.
#    - Alternativa Serverless: AWS App Runner o AWS SageMaker Real-Time Inference Endpoints.
# 2. Balanceo de Carga y Red:
#    - Colocar la API en subredes privadas sin IP pública.
#    - Un Application Load Balancer (ALB) público con certificado SSL/TLS de AWS Certificate Manager (ACM).
# 3. Base de Datos de MLflow:
#    - Amazon RDS PostgreSQL (Multi-AZ) para almacenar metadatos de corridas, métricas y modelos.
# 4. Observabilidad:
#    - AWS CloudWatch Container Insights o Datadog para métricas y logs estructurados.
# =========================================================================
