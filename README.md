# 📊 Quant Trading System

A **Python-based Quantitative Trading System** that:
- Scrapes stock data from **Yahoo Finance (`yfinance`)**
- Backtests trading strategies using **Backtrader**
- Optimizes strategy parameters using **Bayesian Optimization**
- Generates **HTML reports** for backtest results
- Provides **API endpoints (FastAPI)** for frontend integration

## 🔥 Features
✅ **Data Scraper** (Yahoo Finance)  
✅ **Backtester** (Supports multiple trading strategies)  
✅ **Strategy Optimizer** (Using Bayesian Optimization)  
✅ **HTML Report Generation** (For backtest visualization)  
✅ **FastAPI-based API** for serving data  
✅ **PostgreSQL/MongoDB Database Integration**  
✅ **Docker Support for Deployment**  

## 🛠 Tech Stack
- **FastAPI** (Backend API)
- **yfinance** (Data Scraping)
- **Backtrader** (Backtesting)
- **Bayesian Optimization** (Parameter Tuning)
- **PostgreSQL/MongoDB** (Data Storage)
- **Jinja2** (HTML Reporting)
- **Docker** (Containerization)
- **Poetry** (Dependency Management)

## 📂 Project Structure
```
quant_system/
│── src/
│   ├── api/
│   ├── backtester/
│   ├── data_scraper/
│   ├── database/
│   ├── models/
│   ├── optimizer/
│   ├── reports/
│   ├── services/
│   ├── tests/
│   ├── utils/
│── reports_output/
│── pyproject.toml
│── README.md
│── Dockerfile
│── .env
│── .gitignore
````

## 🚀 Installation & Setup

### **1️⃣ Install Poetry**
If you haven't installed **Poetry**, run:
```bash
pip install poetry
```

### **2️⃣ Install Poetry**
```bash
git clone https://github.com/yourusername/quant-system.git
cd quant-system
```

### **3️⃣ Install Dependencies**
```bash
poetry install
```

### **4️⃣ Activate Virtual Environment**
```bash
poetry shell
```

### **5️⃣ Start the FastAPI Server**
```bash
poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```


### **📌 Access API Documentation at:**
```
http://localhost:8000/docs
```

## 🏆 Running a Backtest & Generating Reports
### Backtest a Strategy
### Generate HTML Report
### View Report in Browser

## 📜 API Endpoints

| Method | Endpoint               | Description            |
|--------|------------------------|------------------------|
| GET    | /data/{ticker}         | Fetch stock data       |
| GET    | /backtest/{strategy}   | Run backtest           |
| GET    | /optimize/{strategy}   | Optimize strategy      |
| GET    | /report/{strategy}     | View backtest report   |


## 🧪 Running Tests
```bash
poetry run pytest
```

## 🎯 Linting & Formatting
Ensure code quality by running:
```bash
poetry run black src/
poetry run isort src/
poetry run ruff check src/
```

To automatically fix issues where possible:
```bash
poetry run ruff check --fix src/
```

You can also run all linting tools at once with:
```bash
poetry run black src/ && poetry run isort src/ && poetry run ruff check src/
```

## 🚀 Deploy with Docker

### Build the Docker Image

```bash
docker build -t quant-trading-app .
```

### Run the Container

```bash
docker run -p 8000:8000 quant-trading-app
```

## 🔮 Future Enhancements
- Frontend (React/MERN) Integration
- Live Trading Module (Alpaca API / Interactive Brokers)
- Multi-Asset Portfolio Optimization

## 🤝 Contributions

Feel free to fork and submit PRs! 🚀

## 📜 License

MIT License - Use it freely! 🎯