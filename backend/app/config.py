import os
from pydantic_settings import BaseSettings

class IndustrialMindConfig(BaseSettings):
    # App Config
    app_name: str = "Multimodal Predictive Maintenance API"
    environment: str = "development"
    
    # Model Dimensions
    sensor_input_dim: int = 14
    fused_embed_dim: int = 256
    gnn_hidden_channels: int = 128
    gnn_out_channels: int = 64
    
    # Redis Cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl_seconds: int = 5
    
    # Postgres Vector DB
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "postgres"
    pg_pass: str = "postgres"
    pg_db: str = "industrial_mind"
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.pg_user}:{self.pg_pass}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        
    # Gemini Configuration
    gemini_api_key: str = "MOCK_KEY_FOR_TESTING"
    
    # ML Inference
    enable_fp16: bool = True
    
    # JWT Auth
    jwt_private_key_path: str = ".certs/private_key.pem"
    jwt_public_key_path: str = ".certs/public_key.pem"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Global config instance
settings = IndustrialMindConfig()
