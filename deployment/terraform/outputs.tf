# Project Outputs
output "project_id" {
  description = "The project ID where resources are deployed"
  value       = var.project_id
}

output "region" {
  description = "The deployment region"
  value       = var.region
}

# Network Outputs
output "vpc_id" {
  description = "The ID of the VPC"
  value       = google_compute_network.vpc.id
}

output "subnet_id" {
  description = "The ID of the subnet"
  value       = google_compute_subnetwork.subnet.id
}

output "vpc_name" {
  description = "The name of the VPC"
  value       = google_compute_network.vpc.name
}

# Storage Outputs
output "storage_bucket" {
  description = "The name of the storage bucket"
  value       = google_storage_bucket.model_storage.name
}

output "storage_bucket_url" {
  description = "The URL of the storage bucket"
  value       = google_storage_bucket.model_storage.url
}

# Service Account Outputs
output "compute_service_account" {
  description = "The email of the compute service account"
  value       = google_service_account.compute_sa.email
}

output "prediction_service_account" {
  description = "The email of the prediction service account"
  value       = google_service_account.prediction_sa.email
}

# Compute Instance Outputs
output "compute_instance_name" {
  description = "The name of the compute instance"
  value       = google_compute_instance.clv_instance.name
}

output "compute_instance_ip" {
  description = "The internal IP of the compute instance"
  value       = google_compute_instance.clv_instance.network_interface[0].network_ip
}

# Cloud Run Outputs
output "cloud_run_url" {
  description = "The URL of the Cloud Run service"
  value       = google_cloud_run_service.clv_service.status[0].url
}

output "cloud_run_service_name" {
  description = "The name of the Cloud Run service"
  value       = google_cloud_run_service.clv_service.name
}

# Vertex AI Outputs
output "vertex_ai_endpoint" {
  description = "The Vertex AI endpoint name"
  value       = google_vertex_ai_endpoint.clv_endpoint.name
}

output "vertex_ai_endpoint_id" {
  description = "The Vertex AI endpoint ID"
  value       = google_vertex_ai_endpoint.clv_endpoint.id
}

# KMS Outputs
output "kms_keyring" {
  description = "The KMS keyring name"
  value       = var.security.enable_encryption ? google_kms_key_ring.clv_keyring[0].name : null
}

output "kms_key" {
  description = "The KMS key name"
  value       = var.security.enable_encryption ? google_kms_crypto_key.clv_key[0].name : null
}

# Monitoring Outputs
output "monitoring_alert_policy" {
  description = "The name of the monitoring alert policy"
  value       = google_monitoring_alert_policy.error_rate.name
}

output "notification_channel" {
  description = "The notification channel email"
  value       = var.monitoring.alert_email
}

# Resource URLs
output "resource_urls" {
  description = "URLs for accessing various resources"
  value = {
    cloud_run_service = google_cloud_run_service.clv_service.status[0].url
    storage_bucket    = "gs://${google_storage_bucket.model_storage.name}"
    vertex_ai_endpoint = "https://console.cloud.google.com/vertex-ai/endpoints/${google_vertex_ai_endpoint.clv_endpoint.name}?project=${var.project_id}"
  }
}

# IAM Outputs
output "service_account_roles" {
  description = "Roles assigned to service accounts"
  value = {
    compute = [
      for binding in google_project_iam_member.compute_sa_bindings : binding.role
    ]
    prediction = [
      for binding in google_project_iam_member.prediction_sa_bindings : binding.role
    ]
  }
}

# Environment Info
output "environment_info" {
  description = "Environment deployment information"
  value = {
    environment = var.environment
    project     = var.project_id
    region      = var.region
    labels      = var.labels
  }
}
