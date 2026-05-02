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

    prompter_model: str = "gpt-4o-mini"
    evaluator_model: str = "gpt-4o-mini"
    trajectories_path: Path = Field(
        default=Path("./data/trajectories.jsonl"),
        alias="BRAINSTORMLM_TRAJECTORIES_PATH",
    )

    embedding_model: str = "text-embedding-3-small"
    lesson_extractor_model: str = "gpt-4o-mini"
    memory_dir: Path = Field(default=Path("./data/memory"), alias="BRAINSTORMLM_MEMORY_DIR")
    lessons_path: Path = Field(
        default=Path("./data/memory/lessons.jsonl"),
        alias="BRAINSTORMLM_LESSONS_PATH",
    )

    hf_base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    adapters_dir: Path = Field(
        default=Path("./data/adapters"), alias="BRAINSTORMLM_ADAPTERS_DIR"
    )
    default_adapter_path: Path | None = Field(
        default=None, alias="BRAINSTORMLM_DEFAULT_ADAPTER"
    )
    preferences_path: Path = Field(
        default=Path("./data/preferences.jsonl"),
        alias="BRAINSTORMLM_PREFERENCES_PATH",
    )
    hf_token: SecretStr | None = Field(default=None, alias="HF_TOKEN")
    modal_workspace: str | None = Field(default=None, alias="MODAL_WORKSPACE")

    steering_dir: Path = Field(
        default=Path("./data/steering"), alias="BRAINSTORMLM_STEERING_DIR"
    )
    default_steering_path: Path | None = Field(
        default=None, alias="BRAINSTORMLM_STEERING_PATH"
    )
    default_steering_strength: float = 1.0
    default_steering_layers: tuple[int, ...] | None = None

    # plan-06 (production) — server, human feedback, A/B routing, promotion.
    api_host: str = Field(default="127.0.0.1", alias="BRAINSTORMLM_API_HOST")
    api_port: int = Field(default=8000, alias="BRAINSTORMLM_API_PORT")
    feedback_human_path: Path = Field(
        default=Path("./data/feedback_human.jsonl"),
        alias="BRAINSTORMLM_FEEDBACK_HUMAN_PATH",
    )
    ab_config_path: Path = Field(
        default=Path("./data/ab_config.json"),
        alias="BRAINSTORMLM_AB_CONFIG_PATH",
    )
    promoted_adapter_path: Path = Field(
        default=Path("./data/adapters/promoted.json"),
        alias="BRAINSTORMLM_PROMOTED_ADAPTER_PATH",
    )
    server_log_path: Path = Field(
        default=Path("./data/logs/server.jsonl"),
        alias="BRAINSTORMLM_SERVER_LOG_PATH",
    )
    rate_limit_capacity: int = 30
    rate_limit_refill_per_sec: float = 0.5
    user_prompt_max_chars: int = 2000

    # Corpius 1 (ShroomsLLM) integration
    corpius_config_path: Path = Field(
        default=Path("./data/corpius/corpius_1.json"),
        alias="BRAINSTORMLM_CORPIUS_CONFIG",
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
