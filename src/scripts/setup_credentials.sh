#!/bin/bash
# scripts/setup_credentials.sh

# Create GCP credentials directory
mkdir -p ~/.gcp/credentials
chmod 700 ~/.gcp
chmod 700 ~/.gcp/credentials

# Move credentials file
if [ -f "service-account-key.json" ]; then
    mv service-account-key.json ~/.gcp/credentials/
    chmod 600 ~/.gcp/credentials/service-account-key.json
fi

# Update .env file
echo "GOOGLE_APPLICATION_CREDENTIALS=~/.gcp/credentials/service-account-key.json" >> .env
