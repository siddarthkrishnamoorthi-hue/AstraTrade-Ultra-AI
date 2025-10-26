# AstraTrade-Ultra-AI
Advanced AI-powered MT5 trading system with high-frequency news trading, adaptive learning, and multi-strategy execution.

## 🎯 Key Features
- High-frequency news trading (<100ms reaction time)
- Adaptive learning with real-time strategy optimization
- Advanced technical analysis with SMC/ICT patterns
- LSTM + XGBoost ensemble predictions
- Reinforcement learning for strategy evolution
- Comprehensive risk management

## � Requirements

### Hardware Requirements
- CPU: 8+ cores (Intel i7/Ryzen 7 or better)
- RAM: 32GB DDR4
- Storage: 500GB NVMe SSD
- Network: Low latency (<50ms) connection
- GPU: NVIDIA GPU with 8GB+ VRAM (optional)

### Software Requirements
- Windows 10/11 or Ubuntu 22.04 LTS
- Python 3.10+
- MetaTrader 5
- Redis
- PostgreSQL
- Docker (optional)

## 🚀 Quick Start

### 1. Initial Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y build-essential git curl wget htop

# Install Python 3.10
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# Install PostgreSQL and Redis
sudo apt install -y postgresql redis-server
```

### 2. Project Setup
```bash
# Clone repository
git clone https://github.com/yourusername/AstraTrade-Ultra-AI.git
cd AstraTrade-Ultra-AI

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/historical models/cache logs secure
```

## 📊 Database Setup

### 1. Database Installation
```sql
-- Login to PostgreSQL
sudo -u postgres psql

-- Create database and user
CREATE USER astratrade WITH PASSWORD 'your_strong_password';
CREATE DATABASE astratrade OWNER astratrade;
\c astratrade

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create tables
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    pair VARCHAR(10) NOT NULL,
    signal_type VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    entry_price DECIMAL(10,5) NOT NULL,
    stop_loss DECIMAL(10,5) NOT NULL,
    take_profit DECIMAL(10,5) NOT NULL,
    position_size DECIMAL(10,2) NOT NULL,
    entry_time TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    exit_time TIMESTAMPTZ,
    profit_loss DECIMAL(10,2),
    features_json JSONB,
    account_number VARCHAR(20) NOT NULL,
    risk_percentage DECIMAL(5,4),
    strategy_name VARCHAR(50)
);

-- Create market data hypertable
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    pair VARCHAR(10) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    open DECIMAL(10,5) NOT NULL,
    high DECIMAL(10,5) NOT NULL,
    low DECIMAL(10,5) NOT NULL,
    close DECIMAL(10,5) NOT NULL,
    volume DECIMAL(15,2) NOT NULL
);
SELECT create_hypertable('market_data', 'time');

-- Create indexes
CREATE INDEX idx_trades_pair ON trades(pair);
CREATE INDEX idx_trades_entry_time ON trades(entry_time);
CREATE INDEX idx_market_data_pair ON market_data(pair, time DESC);
```

## 💾 Backup System Setup

### 1. Create Backup Script
```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash

# Configuration
BACKUP_DIR="/path/to/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directories
mkdir -p "$BACKUP_DIR"/{database,models,configs}

# Backup database
pg_dump -U astratrade astratrade | gzip > "$BACKUP_DIR/database/db_$TIMESTAMP.sql.gz"

# Backup models
tar -czf "$BACKUP_DIR/models/models_$TIMESTAMP.tar.gz" ./models/

# Backup configs
tar -czf "$BACKUP_DIR/configs/configs_$TIMESTAMP.tar.gz" ./config/

# Cleanup old backups (keep last 30 days)
find "$BACKUP_DIR" -type f -mtime +30 -delete

echo "Backup completed at $(date)" >> "$BACKUP_DIR/backup.log"
EOF

# Make script executable
chmod +x backup.sh
```

### 2. Setup Automated Backups
```bash
# Add to crontab (runs daily at 1 AM)
(crontab -l 2>/dev/null; echo "0 1 * * * /path/to/backup.sh") | crontab -
```

## 📥 Initial Data Download and Training

### 1. Download Historical Data
```powershell
# Download 3 years of data
python scripts/download_historical.py --pairs EURUSD GBPUSD USDJPY XAUUSD --years 3
```

### 2. Start Initial Training
```powershell
# Start model training
python scripts/train_models.py --mode initial
```

## 🔒 Enhanced Security Setup

### 1. Fund Protection Measures
```python
# Configure maximum drawdown and risk limits
risk_limits = {
    "max_daily_drawdown": 0.02,  # 2% daily drawdown limit
    "max_total_drawdown": 0.05,  # 5% total drawdown limit
    "max_risk_per_trade": 0.005, # 0.5% risk per trade
    "max_position_size": 0.1,    # 0.1 lot maximum
    "max_trades_per_hour": 3,    # Maximum 3 trades per hour
    "emergency_stop_loss": 0.1   # 10% account emergency stop
}
```

### 2. Credential Encryption
```python
from utils.credentials import CredentialsManager

# Initialize with enhanced security
creds = CredentialsManager()

# Encrypt sensitive data with rotating keys
creds.encrypt_credentials({
    "alpha_vantage_key": "YOUR_KEY",
    "telegram_token": "YOUR_TOKEN",
    "telegram_chat_id": "YOUR_CHAT_ID"
}, rotate_key=True)

# Verify encryption
creds.verify_encryption()
```

### 3. Code Protection
```bash
# Set up code integrity monitoring
python utils/code_quality.py --init

# Run security checks
python security/monitor.py --verify-integrity

# Enable real-time monitoring
python security/monitor.py --watch
```

### 4. Security Best Practices

#### Account Protection
- Enable 2FA for all accounts
- Use separate demo/live environments
- Implement circuit breakers
- Monitor for unusual activities
- Regular backup verification
- Encrypted storage for all sensitive data

#### Code Security
- Regular integrity checks
- Automated code quality scans
- Remove unused code/imports
- Security-first coding practices
- Regular dependency updates
- Comprehensive logging

#### Trading Safety
- Multi-level validation
- Risk limits enforcement
- Emergency shutdown capability
- Position monitoring
- Performance tracking
- Automated backup systems

#### System Security
- Network isolation
- Firewall configuration
- Regular security updates
- Access control
- Audit logging
- Intrusion detection

### 5. Monitoring and Alerts

```python
# Set up security monitoring
from security.monitor import SecurityMonitor

monitor = SecurityMonitor(config={
    'alert_thresholds': {
        'balance_change': 0.02,      # Alert on 2% balance change
        'drawdown': 0.05,            # Alert on 5% drawdown
        'trade_frequency': 5,        # Max trades per minute
        'position_size': 0.1,        # Max position size in lots
        'system_memory': 90,         # Memory usage alert threshold
        'cpu_usage': 80              # CPU usage alert threshold
    },
    'notification_channels': ['telegram', 'email', 'log'],
    'backup_frequency': 'hourly'
})
```

## 📞 Support and Monitoring

### System Monitoring
```bash
# Install monitoring tools
sudo apt install -y sysstat nethogs iotop

# Monitor system resources
watch -n 60 'echo "=== System Status ==="; free -h; df -h; uptime'
```

### Database Monitoring
```sql
-- Check database size
SELECT pg_size_pretty(pg_database_size('astratrade'));

-- Monitor table sizes
SELECT relname as "Table",
       pg_size_pretty(pg_total_relation_size(relid)) As "Size"
FROM pg_catalog.pg_statio_user_tables 
ORDER BY pg_total_relation_size(relid) DESC;


MarketRegime:
- id, pair, regime_type, volatility, trend_strength, timestamp
- support_resistance (JSON)

AdaptiveStrategy:
- id, pair, strategy_type, parameters (JSON)
- confidence_score, trades_count, avg_profit
- success_rate, updated_at

LearningHistory:
- id, pair, market_state (JSON), action_taken (JSON)
- outcome (JSON), reward, timestamp

Trades:
- id, pair, type, entry_price, exit_price
- volume, profit, timestamp
- strategy_id (FK)

Signals:
- id, pair, signal_type, confidence
- parameters (JSON), timestamp
- strategy_id (FK)

Models:
- id, name, type, parameters (JSON)
- performance_metrics (JSON)
- last_updated

Config:
- id, category, key, value, updated_at

-- How to Run:
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up credentials
python scripts/setup_credentials.py

# 3. Verify setup
python scripts/verify_credentials.py

# 4. Download historical data
python scripts/download_historical.py

# 5. Launch the main system
python main.py

-- Install infrastructure:
# Install Redis
docker run -d -p 6379:6379 redis

# Install Kafka
docker-compose -f docker/docker-compose.yml up -d

# Install PostgreSQL
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres

-- Run setup scripts:
python scripts/setup_credentials.py  # Setup encrypted credentials
python scripts/verify_credentials.py  # Verify all connections

Core Services (4):

Trading Service: Main trading orchestration
ML Service: Model inference and updates
RL Service: Strategy evolution
Security Service: System monitoring
Agents (3):
Adaptive Agent: Core learning agent
News Trader: Fundamental analysis
HF News Trader: High-frequency news trading
Database Schema (7 tables):
