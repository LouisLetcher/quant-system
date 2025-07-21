-- Initialize Quant Trading System Database
-- This script runs automatically when PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS backtests;
CREATE SCHEMA IF NOT EXISTS portfolios;

-- Market data tables
CREATE TABLE IF NOT EXISTS market_data.price_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume BIGINT,
    data_source VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, timestamp, data_source)
);

-- Backtest results
CREATE TABLE IF NOT EXISTS backtests.results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    symbols TEXT[] NOT NULL,
    strategy VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(20,2) NOT NULL,
    final_value DECIMAL(20,2) NOT NULL,
    total_return DECIMAL(10,4) NOT NULL,
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    calmar_ratio DECIMAL(10,4),
    profit_factor DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    volatility DECIMAL(10,4),
    win_rate DECIMAL(10,4),
    parameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Portfolio configurations
CREATE TABLE IF NOT EXISTS portfolios.configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    symbols TEXT[] NOT NULL,
    weights DECIMAL[] DEFAULT NULL,
    initial_capital DECIMAL(20,2) NOT NULL,
    commission DECIMAL(8,6) DEFAULT 0.001,
    slippage DECIMAL(8,6) DEFAULT 0.002,
    optimization_metric VARCHAR(50) DEFAULT 'sortino_ratio',
    secondary_metrics TEXT[] DEFAULT ARRAY['calmar_ratio', 'sharpe_ratio', 'profit_factor'],
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_price_history_symbol_timestamp ON market_data.price_history(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_price_history_data_source ON market_data.price_history(data_source);
CREATE INDEX IF NOT EXISTS idx_backtest_results_created_at ON backtests.results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy ON backtests.results(strategy);
CREATE INDEX IF NOT EXISTS idx_backtest_results_sortino_ratio ON backtests.results(sortino_ratio DESC);

-- Create update trigger for portfolios
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_portfolios_updated_at 
    BEFORE UPDATE ON portfolios.configurations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default crypto portfolio
INSERT INTO portfolios.configurations (name, symbols, initial_capital, optimization_metric, secondary_metrics, config)
VALUES (
    'Crypto Portfolio',
    ARRAY['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT'],
    10000.00,
    'sortino_ratio',
    ARRAY['calmar_ratio', 'sharpe_ratio', 'profit_factor'],
    '{
        "asset_type": "crypto",
        "data_sources": {
            "primary": ["bybit", "yahoo_finance"],
            "fallback": ["alpha_vantage"]
        },
        "risk_profile": "high",
        "leverage": 1,
        "rebalance_frequency": "weekly"
    }'::jsonb
) ON CONFLICT (name) DO NOTHING;

-- Create user for application (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'quantapp') THEN
        CREATE USER quantapp WITH PASSWORD 'quantapp_secure_password';
    END IF;
END
$$;

-- Grant permissions
GRANT USAGE ON SCHEMA market_data TO quantapp;
GRANT USAGE ON SCHEMA backtests TO quantapp;
GRANT USAGE ON SCHEMA portfolios TO quantapp;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA market_data TO quantapp;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA backtests TO quantapp;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA portfolios TO quantapp;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA market_data TO quantapp;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA backtests TO quantapp;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA portfolios TO quantapp;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Quant Trading System database initialized successfully';
    RAISE NOTICE 'Schemas created: market_data, backtests, portfolios';
    RAISE NOTICE 'Default crypto portfolio added';
    RAISE NOTICE 'Primary metric: sortino_ratio (hedge fund standard)';
END $$;
