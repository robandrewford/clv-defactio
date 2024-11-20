# Project Variables
variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region for resource deployment"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# Compute Engine Variables
variable "compute_instance" {
  description = "Compute Engine instance configuration"
  type = object({
    machine_type = string
    disk_size_gb = number
    disk_type    = string
    gpu_type     = string
    gpu_count    = number
  })
  default = {
    machine_type = "n1-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    gpu_type     = "NVIDIA_TESLA_T4"
    gpu_count    = 1
  }
}

# Cloud Run Variables
variable "cloud_run" {
  description = "Cloud Run service configuration"
  type = object({
    service_name    = string
    min_instances   = number
    max_instances   = number
    cpu             = string
    memory          = string
    timeout_seconds = number
    concurrency     = number
  })
  default = {
    service_name    = "clv-service"
    min_instances   = 1
    max_instances   = 10
    cpu             = "2"
    memory          = "4Gi"
    timeout_seconds = 3600
    concurrency     = 80
  }
}

# Vertex AI Variables
variable "vertex_ai" {
  description = "Vertex AI endpoint configuration"
  type = object({
    endpoint_name     = string
    machine_type     = string
    min_replicas     = number
    max_replicas     = number
    accelerator_type = string
    accelerator_count = number
  })
  default = {
    endpoint_name     = "clv-endpoint"
    machine_type     = "n1-standard-4"
    min_replicas     = 1
    max_replicas     = 5
    accelerator_type = "NVIDIA_TESLA_T4"
    accelerator_count = 1
  }
}

# Network Variables
variable "network" {
  description = "Network configuration"
  type = object({
    vpc_name     = string
    subnet_name  = string
    ip_range     = string
    enable_nat   = bool
  })
  default = {
    vpc_name     = "clv-vpc"
    subnet_name  = "clv-subnet"
    ip_range     = "10.0.0.0/24"
    enable_nat   = true
  }
}

# Storage Variables
variable "storage" {
  description = "Storage configuration"
  type = object({
    bucket_name    = string
    location      = string
    storage_class = string
  })
  default = {
    bucket_name    = "clv-storage"
    location      = "US"
    storage_class = "STANDARD"
  }
}

# Service Account Variables
variable "service_accounts" {
  description = "Service account names"
  type = object({
    compute    = string
    prediction = string
  })
  default = {
    compute    = "clv-compute"
    prediction = "clv-prediction"
  }
}

# Monitoring Variables
variable "monitoring" {
  description = "Monitoring configuration"
  type = object({
    alert_email = string
    metrics     = list(string)
  })
  default = {
    alert_email = "alerts@example.com"
    metrics     = ["prediction_requests", "prediction_latency", "error_rate"]
  }
}

# Tags and Labels
variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {
    environment = "dev"
    project     = "clv-360"
    managed_by  = "terraform"
  }
}

# Security Variables
variable "security" {
  description = "Security configuration"
  type = object({
    enable_encryption = bool
    kms_key_ring     = string
    kms_key_name     = string
  })
  default = {
    enable_encryption = true
    kms_key_ring     = "clv-keyring"
    kms_key_name     = "clv-key"
  }
}

# API Configuration
variable "api" {
  description = "API configuration"
  type = object({
    version              = string
    requests_per_minute = number
    enable_cors         = bool
  })
  default = {
    version              = "v1"
    requests_per_minute = 1000
    enable_cors         = true
  }
}
