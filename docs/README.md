# Quant System Documentation

Complete documentation for the Quantitative Trading System.

## 📚 Documentation Index

### Getting Started
- **[Quick Start](../README.md#quick-start)** - Installation and basic usage
- **[System Commands](cli-guide.md)** - Command-line interface reference

### Development
- **[Development Guide](development.md)** - Setup, testing, and contribution guide
- **[API Reference](api-reference.md)** - Code documentation and examples

### Configuration
- **[Portfolio Configuration](portfolio-config.md)** - Portfolio setup and customization
- **[Data Sources](data-sources.md)** - Supported data providers and setup

### Deployment
- **[Docker Guide](docker.md)** - Containerization and deployment
- **[Production Setup](production.md)** - Production deployment best practices

## 🔧 Configuration Files

All configuration files are located in the `config/` directory:

```
config/
├── portfolios/           # Portfolio configurations
│   ├── crypto.json       # Cryptocurrency portfolio
│   ├── forex.json        # Foreign exchange portfolio
│   ├── stocks_*.json     # Stock portfolios (TraderFox)
│   ├── bonds.json        # Fixed income portfolio
│   ├── commodities.json  # Commodities portfolio
│   └── indices.json      # Index tracking portfolio
└── .env.example          # Environment variables template
```

## 🧪 Testing

The system includes comprehensive test coverage:

- **Unit Tests**: Test individual components
- **Integration Tests**: Test complete workflows
- **Coverage**: Minimum 80% code coverage required

Run tests with:
```bash
pytest                    # All tests
pytest -m "not integration"  # Unit tests only
pytest -m "integration"     # Integration tests only
```

## 🔗 External Links

- **Repository**: https://github.com/LouisLetcher/quant-system
- **Issues**: https://github.com/LouisLetcher/quant-system/issues
- **Releases**: https://github.com/LouisLetcher/quant-system/releases
