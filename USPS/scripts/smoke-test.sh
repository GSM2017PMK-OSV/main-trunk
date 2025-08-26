#!/bin/bash
# USPS Smoke Test Script

set -e

ENVIRONMENT=${1:-staging}
TIMEOUT=300
INTERVAL=10

echo "Running smoke tests for environment: $ENVIRONMENT"

# Function to check service health
check_health() {
    local url="$1"
    local attempt=0
    
    while [ $attempt -lt $((TIMEOUT/INTERVAL)) ]; do
        if curl -f -s "$url" >/dev/null 2>&1; then
            echo "Service at $url is healthy"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep $INTERVAL
    done
    
    echo "Service at $url failed to become healthy"
    return 1
}

# Function to run basic functionality test
run_basic_test() {
    local base_url="$1"
    echo "Running basic functionality test..."
    
    # Test health endpoint
    if ! curl -f "${base_url}/health"; then
        echo "Health check failed"
        return 1
    fi
    
    # Test prediction endpoint
    local test_data='{"system_input": "def test(): return 42", "time_horizon": 10}'
    local response=$(curl -s -X POST "${base_url}/predict" \
        -H "Content-Type: application/json" \
        -d "$test_data")
    
    if ! echo "$response" | jq -e '.prediction' >/dev/null 2>&1; then
        echo "Prediction test failed"
        return 1
    fi
    
    echo "Basic functionality test passed"
    return 0
}

# Main execution
case $ENVIRONMENT in
    staging)
        BASE_URL="http://usps-staging.example.com"
        ;;
    production)
        BASE_URL="https://usps.example.com"
        ;;
    *)
        echo "Unknown environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Run health checks
echo "Checking service health..."
check_health "${BASE_URL}/health"

# Run basic functionality test
echo "Running functionality tests..."
run_basic_test "$BASE_URL"

# Run additional environment-specific tests
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Running production-specific tests..."
    
    # Test SSL certificate
    if ! openssl s_client -connect usps.example.com:443 -servername usps.example.com </dev/null 2>/dev/null | openssl x509 -noout -dates; then
        echo "SSL certificate test failed"
        exit 1
    fi
    
    # Test load balancer
    if ! dig +short usps.example.com | grep -q '.'; then
        echo "DNS resolution test failed"
        exit 1
    fi
fi

echo "All smoke tests passed for $ENVIRONMENT environment"
exit 0
