# backend/app/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Required API keys
    openai_api_key: str
    sendgrid_api_key: str

    # Redis connection (default to localhost if not provided)
    redis_url: str = "redis://localhost:6379/0"

    # Directory where your Chroma/FAISS index is stored
    vectorstore_dir: str = "vectorstore"

    # “From” address for sending emails (must be a verified sender in SendGrid)
    email_sender: str

    # Any application‐specific secret (e.g. for signing JWTs or sessions)
    secret_key: str

    class Config:
        env_file = ".env"
        extra = "ignore"   # Ignore any additional environment variables

settings = Settings()
