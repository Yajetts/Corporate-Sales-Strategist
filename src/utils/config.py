"""Configuration management system for environment variables and model paths"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration class"""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent.parent
    SRC_DIR = BASE_DIR / "src"
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Flask settings
    FLASK_APP = os.getenv("FLASK_APP", "src.api.app")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "5000"))
    API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
    
    # Database settings
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "sales_strategist")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    
    MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
    MONGODB_PORT = int(os.getenv("MONGODB_PORT", "27017"))
    MONGODB_DB = os.getenv("MONGODB_DB", "sales_strategist")
    
    # Redis settings
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    
    # Celery settings
    CELERY_BROKER_URL = os.getenv(
        "CELERY_BROKER_URL",
        f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    )
    CELERY_RESULT_BACKEND = os.getenv(
        "CELERY_RESULT_BACKEND",
        f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    )
    
    # Model paths
    BERT_MODEL_PATH = os.getenv("BERT_MODEL_PATH", str(MODELS_DIR / "bert"))
    AUTOENCODER_MODEL_PATH = os.getenv("AUTOENCODER_MODEL_PATH", str(MODELS_DIR / "autoencoder"))
    GNN_MODEL_PATH = os.getenv("GNN_MODEL_PATH", str(MODELS_DIR / "gnn"))
    RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", str(MODELS_DIR / "rl"))
    LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", str(MODELS_DIR / "lstm"))
    REGRESSION_MODEL_PATH = os.getenv("REGRESSION_MODEL_PATH", str(MODELS_DIR / "regression"))
    
    # MLflow settings
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sales-strategist")
    
    # LLM API settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai or anthropic
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    
    # Model inference settings
    BERT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "512"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    DEVICE = os.getenv("DEVICE", "cuda" if os.path.exists("/proc/driver/nvidia") else "cpu")
    
    # Task timeout settings
    TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", "300"))  # 5 minutes
    LONG_TASK_TIMEOUT = int(os.getenv("LONG_TASK_TIMEOUT", "1800"))  # 30 minutes
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get PostgreSQL database URL"""
        return f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
    
    @classmethod
    def get_mongodb_url(cls) -> str:
        """Get MongoDB connection URL"""
        return f"mongodb://{cls.MONGODB_HOST}:{cls.MONGODB_PORT}/{cls.MONGODB_DB}"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.MODELS_DIR, cls.DATA_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    POSTGRES_DB = "sales_strategist_test"
    MONGODB_DB = "sales_strategist_test"


def get_config(env: Optional[str] = None) -> Config:
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv("FLASK_ENV", "development")
    
    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig,
    }
    
    return config_map.get(env, DevelopmentConfig)
