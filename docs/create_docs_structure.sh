#!/bin/bash

# Create main docs directory
mkdir -p docs

# Create subdirectories
mkdir -p docs/getting-started
mkdir -p docs/user-guide
mkdir -p docs/api-reference
mkdir -p docs/development

# Create empty markdown files
touch docs/index.md

# Getting started files
touch docs/getting-started/installation.md
touch docs/getting-started/configuration.md
touch docs/getting-started/quickstart.md

# User guide files
touch docs/user-guide/data-pipeline.md
touch docs/user-guide/model-training.md
touch docs/user-guide/deployment.md

# API reference files
touch docs/api-reference/models.md
touch docs/api-reference/pipeline.md
touch docs/api-reference/utils.md

# Development files
touch docs/development/contributing.md
touch docs/development/testing.md
touch docs/development/cicd.md

echo "Documentation structure created successfully!" 