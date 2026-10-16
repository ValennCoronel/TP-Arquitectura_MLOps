output "s3_bucket_name" {
  description = "Nombre del Bucket S3 asignado para almacenar los artefactos de MLflow"
  value       = aws_s3_bucket.mlflow_artifacts.id
}

output "s3_bucket_arn" {
  description = "ARN del Bucket S3 para configuración de políticas IAM"
  value       = aws_s3_bucket.mlflow_artifacts.arn
}

output "ec2_public_ip" {
  description = "Dirección IP pública de la instancia EC2 para acceder a la API"
  value       = aws_instance.api_server.public_ip
}

output "api_endpoint_url" {
  description = "URL base sugerida para probar la API una vez desplegada"
  value       = "http://${aws_instance.api_server.public_ip}:8000"
}

output "ec2_instance_id" {
  description = "ID de la instancia EC2 para integración en pipelines de CI/CD (SSM)"
  value       = aws_instance.api_server.id
}

output "ecr_repository_url" {
  description = "URL del repositorio Amazon ECR para push de imagenes Docker"
  value       = aws_ecr_repository.api_repo.repository_url
}

output "ecr_repository_name" {
  description = "Nombre del repositorio ECR"
  value       = aws_ecr_repository.api_repo.name
}
