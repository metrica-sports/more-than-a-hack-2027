terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {}

variable "bucket_name" {
  description = "Nombre globalmente único del bucket para reportes de evaluación"
  type        = string
}

resource "aws_s3_bucket" "evaluation_artifacts" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_public_access_block" "evaluation_artifacts" {
  bucket = aws_s3_bucket.evaluation_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "evaluation_artifacts" {
  bucket = aws_s3_bucket.evaluation_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "evaluation_artifacts" {
  bucket = aws_s3_bucket.evaluation_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}