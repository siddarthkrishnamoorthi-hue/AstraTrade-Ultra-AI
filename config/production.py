"""
Production configuration for AstraTrade Ultra AI
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"

# Create necessary directories
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'astratrade'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

# Redis configuration
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'db': 0
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.getenv('API_PORT', 8000)),
    'workers': int(os.getenv('API_WORKERS', 2)),
    'timeout': 300
}

# Trading configuration
TRADING_CONFIG = {
    'max_positions': int(os.getenv('MAX_POSITIONS', 5)),
    'risk_per_trade': float(os.getenv('RISK_PER_TRADE', 1.0)),
    'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', 3.0)),
    'trading_hours': {
        'start': '00:00',
        'end': '23:59'
    }
}

# Model configuration
MODEL_CONFIG = {
    'batch_size': 32,
    'update_interval': 3600,  # 1 hour
    'min_samples': 100,
    'confidence_threshold': 0.7
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOGS_DIR / 'astratrade.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# Monitoring configuration
MONITOR_CONFIG = {
    'enabled': True,
    'interval': 60,  # seconds
    'metrics': ['cpu', 'memory', 'disk', 'network'],
    'alerts': {
        'cpu_threshold': 80,  # percentage
        'memory_threshold': 85,  # percentage
        'disk_threshold': 90,  # percentage
    }
}

# Security configuration
SECURITY_CONFIG = {
    'api_key_required': True,
    'rate_limit': {
        'requests': 100,
        'period': 60  # seconds
    },
    'allowed_ips': os.getenv('ALLOWED_IPS', '').split(','),
    'ssl_enabled': bool(os.getenv('SSL_ENABLED', False))
}