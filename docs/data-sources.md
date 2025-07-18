# Data Sources Guide

Guide to supported data sources and their configuration.

## Supported Sources

### 1. Yahoo Finance (Free)
- **Assets**: Stocks, ETFs, Indices, Forex, Crypto
- **API Key**: Not required
- **Rate Limits**: Moderate
- **Reliability**: High
- **Symbol Format**: `AAPL`, `EURUSD=X`, `BTC-USD`

### 2. Alpha Vantage
- **Assets**: Stocks, Forex, Crypto, Commodities
- **API Key**: Required (free tier available)
- **Rate Limits**: 5 calls/minute (free), 75 calls/minute (premium)
- **Symbol Format**: `AAPL`, `EUR/USD`, `BTC`

### 3. Twelve Data
- **Assets**: Stocks, Forex, Crypto, ETFs
- **API Key**: Required
- **Rate Limits**: 800 calls/day (free)
- **Symbol Format**: `AAPL`, `EUR/USD`, `BTC/USD`

### 4. Polygon.io
- **Assets**: Stocks, Options, Forex, Crypto
- **API Key**: Required
- **Rate Limits**: Based on plan
- **Symbol Format**: `AAPL`, `C:EURUSD`, `X:BTCUSD`

### 5. Tiingo
- **Assets**: Stocks, ETFs, Forex, Crypto
- **API Key**: Required
- **Rate Limits**: 1000 calls/hour (free)
- **Symbol Format**: `AAPL`, `EURUSD`, `BTCUSD`

### 6. Finnhub
- **Assets**: Stocks, Forex, Crypto
- **API Key**: Required
- **Rate Limits**: 60 calls/minute (free)
- **Symbol Format**: `AAPL`, `OANDA:EUR_USD`, `BINANCE:BTCUSDT`

### 7. Bybit
- **Assets**: Crypto derivatives
- **API Key**: Optional (public data)
- **Rate Limits**: High
- **Symbol Format**: `BTCUSDT`, `ETHUSDT`

### 8. Pandas DataReader
- **Assets**: Economic data (FRED, World Bank, etc.)
- **API Key**: Not required
- **Symbol Format**: `GDP`, `UNRATE`

## Configuration

### Environment Variables
Create a `.env` file:
```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_key_here
TWELVE_DATA_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
TIINGO_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here

# Optional Bybit API (for private data)
BYBIT_API_KEY=your_key_here
BYBIT_API_SECRET=your_secret_here
```

### Portfolio Configuration
Specify data sources in portfolio configs:
```json
{
  "data_source": {
    "primary_source": "yahoo",
    "fallback_sources": ["alpha_vantage", "twelve_data"]
  }
}
```

## Symbol Transformation

The system automatically transforms symbols between different data source formats:

| Asset Type | Yahoo Finance | Alpha Vantage | Twelve Data | Bybit |
|------------|---------------|---------------|-------------|-------|
| **Stocks** | `AAPL` | `AAPL` | `AAPL` | N/A |
| **Forex** | `EURUSD=X` | `EUR/USD` | `EUR/USD` | N/A |
| **Crypto** | `BTC-USD` | `BTC` | `BTC/USD` | `BTCUSDT` |
| **Indices** | `^GSPC` | `SPX` | `SPX` | N/A |

## Best Practices

### 1. Use Fallback Sources
Always configure fallback sources for reliability:
```json
{
  "primary_source": "yahoo",
  "fallback_sources": ["alpha_vantage", "twelve_data"]
}
```

### 2. Respect Rate Limits
- Use caching to minimize API calls
- Implement delays between requests
- Monitor usage for paid services

### 3. Data Quality
- Validate data after fetching
- Check for missing values
- Compare across sources for consistency

### 4. Cost Management
- Use free sources (Yahoo Finance) when possible
- Monitor API usage for paid services
- Cache data to reduce API calls

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   # Check environment variables
   echo $ALPHA_VANTAGE_API_KEY

   # Verify .env file
   cat .env
   ```

2. **Rate Limit Exceeded**
   ```bash
   # Clear cache and retry later
   python -m src.cli.unified_cli cache clear --all
   ```

3. **Symbol Not Found**
   ```bash
   # Check symbol format for the data source
   # Use data validation command
   python -m src.cli.unified_cli data validate --symbol AAPL
   ```

4. **Network Issues**
   ```bash
   # Test connectivity
   ping finance.yahoo.com

   # Check firewall/proxy settings
   ```

### Debug Mode
Enable debug logging for detailed information:
```bash
export LOG_LEVEL=DEBUG
python -m src.cli.unified_cli data download --symbols AAPL
```

## Getting API Keys

### Alpha Vantage
1. Visit https://www.alphavantage.co/support/#api-key
2. Sign up for free account
3. Get API key from dashboard

### Twelve Data
1. Visit https://twelvedata.com/pricing
2. Sign up for free plan
3. Get API key from account settings

### Polygon.io
1. Visit https://polygon.io/pricing
2. Sign up for plan
3. Get API key from dashboard

### Tiingo
1. Visit https://api.tiingo.com/
2. Sign up for free account
3. Get API token from account

### Finnhub
1. Visit https://finnhub.io/pricing
2. Sign up for free account
3. Get API key from dashboard

## Performance Optimization

### Caching Strategy
- Cache data for 1 hour (default)
- Use Parquet format for compression
- Implement cache expiration

### Parallel Downloads
- Fetch multiple symbols concurrently
- Use connection pooling
- Implement retry logic

### Data Validation
- Check data completeness
- Validate OHLCV format
- Remove invalid entries
