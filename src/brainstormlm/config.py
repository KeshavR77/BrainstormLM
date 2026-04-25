from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    together_api_key: SecretStr = Field(alias="TOGETHER_API_KEY")

    data_dir: Path = Field(default=Path("./data"), alias="BRAINSTORMLM_DATA_DIR")
    cache_dir: Path = Field(default=Path("./data/cache"), alias="BRAINSTORMLM_CACHE_DIR")

    together_base_model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo"
    default_temperature: float = 1.0
    default_max_tokens: int = 1024
    default_top_p: float = 0.95


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
