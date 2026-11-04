from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  # Database
  database_name: str = "history.db"

  # OpenAI
  openai_api_key: str = "xxxxx"

  # AI
  ai_model: str = "gpt-4o-mini"
  ai_model_temperature: float = 0.0

  # Type-Safe API
  type_safe_api_key: str = "xxxxx"
  type_safe_model: str = "jev-latest"

  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
  return Settings()
