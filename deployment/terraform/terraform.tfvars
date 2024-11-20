# Project Settings
project_id  = "your-project-id"
region      = "us-central1"
environment = "dev"  # or "staging" or "prod"

# Compute Instance Settings
compute_instance = {
  machine_type = "n1-standard-4"
  disk_size_gb = 100
  disk_type    = "pd-ssd"
  gpu_type     = "NVIDIA_TESLA_T4"
  gpu_count    = 1
}

# Cloud Run Settings
cloud_run = {
  service_name    = "clv-service"
  min_instances   = 1
  max_instances   = 10
  cpu             = "2"
  memory          = "4Gi"
  timeout_seconds = 3600
  concurrency     = 80
}

# Vertex AI Settings
vertex_ai = {
  endpoint_name     = "clv-endpoint"
  machine_type     = "n1-standard-4"
  min_replicas     = 1
  max_replicas     = 5
  accelerator_type = "NVIDIA_TESLA_T4"
  accelerator_count = 1
}

# Network Settings
network = {
  vpc_name    = "clv-vpc"
  subnet_name = "clv-subnet"
  ip_range    = "10.0.0.0/24"
  enable_nat  = true
}

# Storage Settings
storage = {
  bucket_name   = "clv-storage-bucket"
  location      = "US"
  storage_class = "STANDARD"
}

# Service Account Settings
service_accounts = {
  compute    = "clv-compute-sa"
  prediction = "clv-prediction-sa"
}

# Monitoring Settings
monitoring = {
  alert_email = "your-team@company.com"
  metrics     = [
    "prediction_requests",
    "prediction_latency",
    "error_rate"
  ]
}

# Labels
labels = {
  environment = "dev"
  project     = "clv-360"
  managed_by  = "terraform"
  team        = "ml-ops"
}

# Security Settings
security = {
  enable_encryption = true
  kms_key_ring     = "clv-keyring"
  kms_key_name     = "clv-key"
}

# API Settings
api = {
  version              = "v1"
  requests_per_minute = 1000
  enable_cors         = true
}
