-- Initialize database for quant system
-- This script sets up the basic database schema for PostgreSQL

-- Create database if it doesn't exist (handled by docker-entrypoint)
-- CREATE DATABASE IF NOT EXISTS quant_system;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS system;

-- Create trading tables
CREATE TABLE IF NOT EXISTS trading.symbols (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(255),
    asset_type VARCHAR(50) NOT NULL CHECK (asset_type IN ('stocks', 'crypto', 'forex', 'futures')),
    exchange VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trading.strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    parameters JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trading.portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    risk_profile VARCHAR(50) CHECK (risk_profile IN ('conservative', 'moderate', 'aggressive')),
    target_return DECIMAL(5,4),
    symbols TEXT[], -- Array of symbol IDs
    strategies TEXT[], -- Array of strategy IDs
    allocation JSONB, -- Symbol allocations
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create analytics tables
CREATE TABLE IF NOT EXISTS analytics.backtest_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol_id UUID REFERENCES trading.symbols(id),
    strategy_id UUID REFERENCES trading.strategies(id),
    portfolio_id UUID REFERENCES trading.portfolios(id),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,2),
    total_return DECIMAL(8,4),
    annualized_return DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    volatility DECIMAL(8,4),
    trades_count INTEGER,
    win_rate DECIMAL(5,4),
    parameters JSONB,
    results JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analytics.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    backtest_run_id UUID REFERENCES analytics.backtest_runs(id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6),
    metric_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create system tables
CREATE TABLE IF NOT EXISTS system.cache_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cache_key VARCHAR(255) NOT NULL UNIQUE,
    cache_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    data_type VARCHAR(50),
    size_bytes BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    file_path TEXT
);

CREATE TABLE IF NOT EXISTS system.api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    response_time_ms INTEGER,
    status_code INTEGER,
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON trading.symbols(symbol);
CREATE INDEX IF NOT EXISTS idx_symbols_asset_type ON trading.symbols(asset_type);
CREATE INDEX IF NOT EXISTS idx_strategies_name ON trading.strategies(name);
CREATE INDEX IF NOT EXISTS idx_portfolios_name ON trading.portfolios(name);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_symbol_strategy ON analytics.backtest_runs(symbol_id, strategy_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_dates ON analytics.backtest_runs(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_backtest ON analytics.performance_metrics(backtest_run_id);
CREATE INDEX IF NOT EXISTS idx_cache_metadata_key ON system.cache_metadata(cache_key);
CREATE INDEX IF NOT EXISTS idx_cache_metadata_type ON system.cache_metadata(cache_type);
CREATE INDEX IF NOT EXISTS idx_cache_metadata_expires ON system.cache_metadata(expires_at);
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON system.api_usage(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_usage_created ON system.api_usage(created_at);

-- Create functions for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_symbols_updated_at BEFORE UPDATE ON trading.symbols FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON trading.strategies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_portfolios_updated_at BEFORE UPDATE ON trading.portfolios FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert some default data
INSERT INTO trading.strategies (name, description, parameters) VALUES
    ('rsi', 'Relative Strength Index strategy', '{"period": 14, "overbought": 70, "oversold": 30}'),
    ('macd', 'MACD strategy', '{"fast": 12, "slow": 26, "signal": 9}'),
    ('sma_crossover', 'Simple Moving Average Crossover', '{"short_window": 50, "long_window": 200}'),
    ('bollinger_bands', 'Bollinger Bands strategy', '{"period": 20, "std_dev": 2}')
ON CONFLICT (name) DO NOTHING;

-- Insert some default symbols
INSERT INTO trading.symbols (symbol, name, asset_type, exchange) VALUES
    ('AAPL', 'Apple Inc.', 'stocks', 'NASDAQ'),
    ('MSFT', 'Microsoft Corporation', 'stocks', 'NASDAQ'),
    ('GOOGL', 'Alphabet Inc.', 'stocks', 'NASDAQ'),
    ('AMZN', 'Amazon.com Inc.', 'stocks', 'NASDAQ'),
    ('TSLA', 'Tesla Inc.', 'stocks', 'NASDAQ'),
    ('BTCUSDT', 'Bitcoin/USDT', 'crypto', 'Binance'),
    ('ETHUSDT', 'Ethereum/USDT', 'crypto', 'Binance'),
    ('BNBUSDT', 'BNB/USDT', 'crypto', 'Binance'),
    ('EURUSD=X', 'EUR/USD', 'forex', 'FOREX'),
    ('GBPUSD=X', 'GBP/USD', 'forex', 'FOREX'),
    ('USDJPY=X', 'USD/JPY', 'forex', 'FOREX')
ON CONFLICT (symbol) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA trading TO quantuser;
GRANT USAGE ON SCHEMA analytics TO quantuser;
GRANT USAGE ON SCHEMA system TO quantuser;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA trading TO quantuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA analytics TO quantuser;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA system TO quantuser;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA trading TO quantuser;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analytics TO quantuser;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA system TO quantuser;

-- Create views for common queries
CREATE OR REPLACE VIEW analytics.portfolio_performance AS
SELECT
    p.name as portfolio_name,
    COUNT(br.id) as backtest_count,
    AVG(br.total_return) as avg_return,
    AVG(br.sharpe_ratio) as avg_sharpe_ratio,
    AVG(br.max_drawdown) as avg_max_drawdown,
    MAX(br.created_at) as last_backtest
FROM trading.portfolios p
LEFT JOIN analytics.backtest_runs br ON p.id = br.portfolio_id
WHERE p.is_active = true
GROUP BY p.id, p.name;

CREATE OR REPLACE VIEW analytics.strategy_performance AS
SELECT
    s.name as strategy_name,
    COUNT(br.id) as backtest_count,
    AVG(br.total_return) as avg_return,
    AVG(br.sharpe_ratio) as avg_sharpe_ratio,
    AVG(br.win_rate) as avg_win_rate,
    MAX(br.created_at) as last_backtest
FROM trading.strategies s
LEFT JOIN analytics.backtest_runs br ON s.id = br.strategy_id
WHERE s.is_active = true
GROUP BY s.id, s.name;

-- Create materialized view for performance dashboard
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.daily_performance_summary AS
SELECT
    DATE(br.created_at) as backtest_date,
    COUNT(*) as total_backtests,
    COUNT(DISTINCT br.symbol_id) as unique_symbols,
    COUNT(DISTINCT br.strategy_id) as unique_strategies,
    AVG(br.total_return) as avg_return,
    STDDEV(br.total_return) as return_volatility,
    AVG(br.sharpe_ratio) as avg_sharpe_ratio,
    COUNT(CASE WHEN br.total_return > 0 THEN 1 END)::FLOAT / COUNT(*) as win_rate
FROM analytics.backtest_runs br
GROUP BY DATE(br.created_at)
ORDER BY backtest_date DESC;

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_performance_summary_date ON analytics.daily_performance_summary(backtest_date);

-- Refresh materialized view (will be empty initially)
REFRESH MATERIALIZED VIEW analytics.daily_performance_summary;

-- Create function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW analytics.daily_performance_summary;
END;
$$ LANGUAGE plpgsql;

COMMIT;
