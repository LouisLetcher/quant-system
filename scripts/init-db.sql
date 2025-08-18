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

-- Legacy backtest results (backward compatibility)
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

-- Best strategies per asset (curated data)
CREATE TABLE IF NOT EXISTS backtests.best_strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    best_strategy VARCHAR(100) NOT NULL,

    -- Best performance metrics
    sortino_ratio DECIMAL(10,4) NOT NULL,
    sharpe_ratio DECIMAL(10,4),
    calmar_ratio DECIMAL(10,4),
    profit_factor DECIMAL(10,4),
    total_return DECIMAL(10,4) NOT NULL,
    max_drawdown DECIMAL(10,4),
    volatility DECIMAL(10,4),
    win_rate DECIMAL(10,4),
    num_trades INTEGER,

    -- Risk metrics for recommendations
    risk_score DECIMAL(10,4),
    correlation_group VARCHAR(50),

    -- Trading parameters for position sizing
    risk_per_trade DECIMAL(10,4),
    stop_loss_pct DECIMAL(10,4),
    take_profit_pct DECIMAL(10,4),

    -- Best strategy parameters
    best_parameters JSONB,

    -- Metadata
    backtest_run_id VARCHAR(100) NOT NULL,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(symbol, timeframe)
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

-- All optimization results (historical record of every optimization run)
CREATE TABLE IF NOT EXISTS backtests.all_optimization_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    strategy VARCHAR(100) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,

    -- Parameter combination being tested
    parameters JSONB NOT NULL,

    -- Performance metrics for this parameter set
    sortino_ratio DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    calmar_ratio DECIMAL(10,4),
    profit_factor DECIMAL(10,4),
    total_return DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    volatility DECIMAL(10,4),
    win_rate DECIMAL(10,4),
    num_trades INTEGER,

    -- Optimization metadata
    iteration_number INTEGER,
    optimization_metric VARCHAR(50) DEFAULT 'sortino_ratio',
    portfolio_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(run_id, symbol, strategy, timeframe, iteration_number)
);

-- Best optimization results per asset/strategy (curated data)
CREATE TABLE IF NOT EXISTS backtests.best_optimization_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    strategy VARCHAR(100) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,

    -- Best performance metrics
    best_sortino_ratio DECIMAL(10,4) NOT NULL,
    best_sharpe_ratio DECIMAL(10,4),
    best_calmar_ratio DECIMAL(10,4),
    best_profit_factor DECIMAL(10,4),
    best_total_return DECIMAL(10,4),
    best_max_drawdown DECIMAL(10,4),
    best_volatility DECIMAL(10,4),
    best_win_rate DECIMAL(10,4),
    best_num_trades INTEGER,

    -- Optimization summary
    best_parameters JSONB NOT NULL,
    optimization_metric VARCHAR(50) DEFAULT 'sortino_ratio',
    total_iterations INTEGER,
    optimization_time_seconds DECIMAL(10,2),
    parameter_ranges JSONB,

    -- Metadata
    optimization_run_id VARCHAR(100) NOT NULL,
    portfolio_name VARCHAR(255),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(symbol, strategy, timeframe)
);

-- Legacy optimization results (backward compatibility)
CREATE TABLE IF NOT EXISTS backtests.optimization_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES portfolios.configurations(id),
    strategy VARCHAR(100) NOT NULL,
    optimization_metric VARCHAR(50) DEFAULT 'sortino_ratio',
    best_sortino_ratio DECIMAL(10,4),
    best_calmar_ratio DECIMAL(10,4),
    best_sharpe_ratio DECIMAL(10,4),
    best_profit_factor DECIMAL(10,4),
    best_parameters JSONB,
    iterations INTEGER,
    optimization_time DECIMAL(10,2),
    parameter_ranges JSONB,
    results_summary JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI recommendations (portfolio-level insights)
CREATE TABLE IF NOT EXISTS backtests.ai_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_name VARCHAR(255) NOT NULL,
    quarter VARCHAR(10) NOT NULL,
    year INTEGER NOT NULL,
    risk_tolerance VARCHAR(20) NOT NULL,
    total_score DECIMAL(10,4),
    confidence DECIMAL(10,4),
    diversification_score DECIMAL(10,4),
    total_assets INTEGER,
    expected_return DECIMAL(10,4),
    portfolio_risk DECIMAL(10,4),
    overall_reasoning TEXT,
    warnings TEXT[],
    correlation_analysis JSONB,
    llm_model VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(portfolio_name, quarter, year, risk_tolerance)
);

-- Individual asset recommendations
CREATE TABLE IF NOT EXISTS backtests.asset_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ai_recommendation_id UUID NOT NULL REFERENCES backtests.ai_recommendations(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    allocation_percentage DECIMAL(10,4) NOT NULL,
    confidence_score DECIMAL(10,4) NOT NULL,
    performance_score DECIMAL(10,4) NOT NULL,
    risk_score DECIMAL(10,4) NOT NULL,
    reasoning TEXT,
    red_flags TEXT[],
    risk_per_trade DECIMAL(10,4),
    stop_loss_pct DECIMAL(10,4),
    take_profit_pct DECIMAL(10,4),
    position_size_usd DECIMAL(20,2),
    best_strategy_id UUID REFERENCES backtests.best_strategies(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(ai_recommendation_id, symbol)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_price_history_symbol_timestamp ON market_data.price_history(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_price_history_data_source ON market_data.price_history(data_source);

-- Legacy results indexes
CREATE INDEX IF NOT EXISTS idx_backtest_results_created_at ON backtests.results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_backtest_results_strategy ON backtests.results(strategy);
CREATE INDEX IF NOT EXISTS idx_backtest_results_sortino_ratio ON backtests.results(sortino_ratio DESC);

-- Best strategies indexes
CREATE INDEX IF NOT EXISTS idx_best_strategies_symbol ON backtests.best_strategies(symbol);
CREATE INDEX IF NOT EXISTS idx_best_strategies_sortino ON backtests.best_strategies(sortino_ratio DESC);
CREATE INDEX IF NOT EXISTS idx_best_strategies_risk ON backtests.best_strategies(risk_score);

-- All optimization results indexes
CREATE INDEX IF NOT EXISTS idx_all_opt_results_run_id ON backtests.all_optimization_results(run_id);
CREATE INDEX IF NOT EXISTS idx_all_opt_results_symbol_strategy ON backtests.all_optimization_results(symbol, strategy);
CREATE INDEX IF NOT EXISTS idx_all_opt_results_sortino ON backtests.all_optimization_results(sortino_ratio DESC);

-- Best optimization results indexes
CREATE INDEX IF NOT EXISTS idx_best_opt_results_symbol ON backtests.best_optimization_results(symbol);
CREATE INDEX IF NOT EXISTS idx_best_opt_results_strategy ON backtests.best_optimization_results(symbol, strategy);
CREATE INDEX IF NOT EXISTS idx_best_opt_results_sortino ON backtests.best_optimization_results(best_sortino_ratio DESC);

-- Legacy optimization results indexes
CREATE INDEX IF NOT EXISTS idx_optimization_results_created_at ON backtests.optimization_results(created_at DESC);

-- AI recommendations indexes
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_created_at ON backtests.ai_recommendations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_portfolio ON backtests.ai_recommendations(portfolio_name);
CREATE INDEX IF NOT EXISTS idx_ai_recommendations_quarter ON backtests.ai_recommendations(quarter, year);

-- Asset recommendations indexes
CREATE INDEX IF NOT EXISTS idx_asset_recommendations_ai_id ON backtests.asset_recommendations(ai_recommendation_id);
CREATE INDEX IF NOT EXISTS idx_asset_recommendations_symbol ON backtests.asset_recommendations(symbol);
CREATE INDEX IF NOT EXISTS idx_asset_recommendations_allocation ON backtests.asset_recommendations(allocation_percentage DESC);

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
