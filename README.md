# Quant System

A unified, Dockerized quantitative backtesting and reporting system. Run cross‑strategy comparisons for asset collections (e.g., bonds) and persist results to PostgreSQL with exportable artifacts.

## 🚀 Quick Start

### Docker Setup

```bash
# Clone repository
git clone <repository-url>
cd quant-system

# Start PostgreSQL and pgAdmin
docker compose up -d postgres pgadmin

# Build the app image (uses DOCKERFILE)
docker compose build quant

# Show CLI help
docker compose run --rm quant python -m src.cli.unified_cli --help

# Interactive shell inside the app container
docker compose run --rm quant bash
```

## 📈 Usage

See also: docs/pgadmin-and-performance.md for DB inspection and performance tips.

The unified CLI currently exposes a single subcommand: `collection`.

### Run Bonds (1d interval, max period, all strategies)

Use the collection key (`bonds`) or the JSON file path. The `direct` action runs the backtests and writes results to the DB. Add `--exports all` to generate CSV/HTML/TradingView artifacts when possible.

```bash
# Using the collection key (recommended)
docker compose run --rm \
  -e STRATEGIES_PATH=/app/external_strategies \
  quant python -m src.cli.unified_cli collection bonds \
  --action direct \
  --interval 1d \
  --period max \
  --strategies all \
  --exports all \
  --log-level INFO

# Using the JSON file
docker compose run --rm \
  -e STRATEGIES_PATH=/app/external_strategies \
  quant python -m src.cli.unified_cli collection config/collections/bonds.json \
  --action direct \
  --interval 1d \
  --period max \
  --strategies all \
  --exports all \
  --log-level INFO
```

Notes

- Default metric is `sortino_ratio`.
- Strategies are mounted at `/app/external_strategies` via `docker-compose.yml`; `STRATEGIES_PATH` makes discovery explicit.
- Artifacts are written under `artifacts/run_*`. DB tables used include `runs`, `backtest_results`, `best_strategies`, and `run_artifacts`.
- pgAdmin is available at `http://localhost:5050` (defaults configured via `.env`/`.env.example`).

### Dry Run (plan only + optional exports)

```bash
docker compose run --rm \
  -e STRATEGIES_PATH=/app/external_strategies \
  quant python -m src.cli.unified_cli collection bonds \
  --interval 1d --period max --strategies all \
  --dry-run --exports all --log-level DEBUG
```

### Other Actions

The `collection` subcommand supports these `--action` values: `backtest`, `direct`, `optimization`, `export`, `report`, `tradingview`. In most workflows, use `--action direct` and optionally `--exports`.

## 🔧 Configuration

### Environment Variables (.env)

```bash
# PostgreSQL (inside the container, use the service name 'postgres')
DATABASE_URL=postgresql://quantuser:quantpass@postgres:5432/quant_system

# Optional data providers
ALPHA_VANTAGE_API_KEY=your_key
TWELVE_DATA_API_KEY=your_key
POLYGON_API_KEY=your_key
TIINGO_API_KEY=your_key
FINNHUB_API_KEY=your_key
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
BYBIT_TESTNET=false

# Optional LLMs
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o
ANTHROPIC_API_KEY=your_key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

Host access tips

- Postgres is published on `localhost:5433` (mapped to container `5432`).
- pgAdmin runs at `http://localhost:5050` (see `.env` for credentials).

### Collections

Sample collections live under `config/collections/`, including `bonds.json` with common bond ETFs.

## 🧪 Testing

```bash
# Run tests in Docker
docker compose run --rm quant pytest
```

## 📊 Exports & Reporting

Artifacts and exports are written under `artifacts/run_*` and `exports/`. When running with `--action direct`, pass `--exports csv,json,report,tradingview` or `--exports all`.

```bash
# Produce exports from DB for bonds without re-running backtests
docker compose run --rm quant \
  python -m src.cli.unified_cli collection bonds --dry-run --exports all
```

## 📚 Further Docs

- docs/pgadmin-and-performance.md — pgAdmin queries and performance tips
- docs/data-sources.md — supported providers and configuration
- docs/development.md — local dev, testing, and repo layout
- docs/docker.md — legacy Docker notes (see README for up-to-date compose)
- docs/features.md — feature overview; some CLI examples are legacy
- docs/cli-guide.md — legacy CLI reference; see README examples for current usage

## 🛠️ Troubleshooting

- Command name: use `docker compose` (or legacy `docker-compose`) consistently.
- Subcommand: it is `collection` (singular), not `collections`.
- Strategy discovery: ensure strategies are mounted at `/app/external_strategies` and set `STRATEGIES_PATH=/app/external_strategies` when running.
- Database URL: inside containers use `postgres:5432` (`DATABASE_URL=postgresql://quantuser:quantpass@postgres:5432/quant_system`). On the host, Postgres is published at `localhost:5433`.
- Initialize tables: if tables are missing, run:
  `docker compose run --rm quant python -c "from src.database.unified_models import create_tables; create_tables()"`
- Long runs/timeouts: backtests can take minutes to hours depending on strategies and symbols. Prefer `--log-level INFO` or `DEBUG` to monitor progress. Use `--dry-run` to validate plans quickly. Extra tips in docs/pgadmin-and-performance.md.
- Permissions/cache: ensure `cache/`, `exports/`, `logs/`, and `artifacts/` exist and are writable on the host (compose mounts them into the container).
- API limits: some data sources rate-limit; providing API keys in `.env` can reduce throttling.

## ⚠️ Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Use at your own risk and always perform your own due diligence before making investment decisions.
