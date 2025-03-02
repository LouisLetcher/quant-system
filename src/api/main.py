from fastapi import FastAPI
from api.routes import data_routes, backtest_routes, optimization_routes, report_routes

app = FastAPI(title="Quant Trading System", version="1.0")

# Include routers
app.include_router(data_routes.router)
app.include_router(backtest_routes.router)
app.include_router(optimization_routes.router)
app.include_router(report_routes.router)

@app.get("/")
def root():
    return {"message": "Quant Trading System API is running!"}