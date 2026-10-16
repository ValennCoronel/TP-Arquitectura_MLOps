variable "aws_region" {
  description = "Región de AWS donde se desplegarán los recursos"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Identificador del proyecto para nombrado y etiquetado de recursos"
  type        = string
  default     = "mlops-churn-poc"
}

variable "environment" {
  description = "Entorno de ejecución (poc, dev, staging, prod)"
  type        = string
  default     = "poc"
}

variable "instance_type" {
  description = "Tipo de instancia EC2 (t3.micro elegible para capa gratuita de AWS)"
  type        = string
  default     = "t3.micro"
}

variable "ssh_allowed_cidr" {
  description = "Rango CIDR autorizado para acceso SSH a la instancia EC2"
  type        = string
  default     = "0.0.0.0/0" # En producción: restringir estrictamente a la VPN corporativa o IP del operador
}
