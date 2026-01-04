from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Supabase Configuration ---
    SUPABASE_URL: str = Field(..., description="The Project URL from your Supabase dashboard")
    SUPABASE_KEY: str = Field(..., description="The Anon/Public Key from Supabase")

    # --- Google GenAI Configuration ---
    GOOGLE_API_KEY: str = Field(..., description="Your Google AI Studio API Key (Gemini)")

    # --- Application Settings ---
    APP_NAME: str = "Life Coach RAG"
    DEBUG: bool = False

    # Pydantic Settings V2 configuration
    # This tells Pydantic to look for a .env file and ignore extra variables
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Create a singleton instance to be imported elsewhere
settings = Settings()
