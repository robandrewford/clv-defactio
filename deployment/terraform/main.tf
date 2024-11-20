# Provider Configuration
provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable Required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "containerregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "aiplatform.googleapis.com"
  ])

  service = each.key
  disable_on_destroy = false
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = var.network.vpc_name
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = var.network.subnet_name
  ip_cidr_range = var.network.ip_range
  network       = google_compute_network.vpc.id
  region        = var.region
}

# Cloud Storage Bucket
resource "google_storage_bucket" "model_storage" {
  name          = "${var.storage.bucket_name}-${var.project_id}"
  location      = var.storage.location
  storage_class = var.storage.storage_class

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  labels = var.labels
}

# Service Accounts
resource "google_service_account" "compute_sa" {
  account_id   = var.service_accounts.compute
  display_name = "CLV Compute Service Account"
}

resource "google_service_account" "prediction_sa" {
  account_id   = var.service_accounts.prediction
  display_name = "CLV Prediction Service Account"
}

# Compute Instance
resource "google_compute_instance" "clv_instance" {
  name         = "clv-compute-${var.environment}"
  machine_type = var.compute_instance.machine_type
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-10"
      size  = var.compute_instance.disk_size_gb
      type  = var.compute_instance.disk_type
    }
  }

  guest_accelerator {
    type  = var.compute_instance.gpu_type
    count = var.compute_instance.gpu_count
  }

  network_interface {
    subnetwork = google_compute_subnetwork.subnet.id
  }

  service_account {
    email  = google_service_account.compute_sa.email
    scopes = ["cloud-platform"]
  }

  labels = var.labels
}

# Cloud Run Service
resource "google_cloud_run_service" "clv_service" {
  name     = var.cloud_run.service_name
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/clv-service:latest"

        resources {
          limits = {
            cpu    = var.cloud_run.cpu
            memory = var.cloud_run.memory
          }
        }
      }

      service_account_name = google_service_account.prediction_sa.email
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = var.cloud_run.min_instances
        "autoscaling.knative.dev/maxScale" = var.cloud_run.max_instances
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Vertex AI Endpoint
resource "google_vertex_ai_endpoint" "clv_endpoint" {
  name         = var.vertex_ai.endpoint_name
  description  = "CLV prediction endpoint"
  location     = var.region

  network      = google_compute_network.vpc.id

  labels       = var.labels
}

# Monitoring
resource "google_monitoring_alert_policy" "error_rate" {
  display_name = "CLV Error Rate Alert"

  conditions {
    display_name = "Error Rate > 1%"

    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/clv/error_rate\""
      duration        = "300s"
      comparison     = "COMPARISON_GT"
      threshold_value = 0.01
    }
  }

  notification_channels = [
    google_monitoring_notification_channel.email.name
  ]
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "CLV Alert Email"
  type         = "email"

  labels = {
    email_address = var.monitoring.alert_email
  }
}

# KMS Configuration (if encryption enabled)
resource "google_kms_key_ring" "clv_keyring" {
  count    = var.security.enable_encryption ? 1 : 0
  name     = var.security.kms_key_ring
  location = "global"
}

resource "google_kms_crypto_key" "clv_key" {
  count           = var.security.enable_encryption ? 1 : 0
  name            = var.security.kms_key_name
  key_ring        = google_kms_key_ring.clv_keyring[0].id
  rotation_period = "7776000s" # 90 days
}

# IA
