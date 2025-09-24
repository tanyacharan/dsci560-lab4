# Polygon.io XGBoost HFT Trading System

A sophisticated high-frequency trading system that combines XGBoost machine learning with technical analysis for automated stock trading using Polygon.io stock market data.

## Features

### Core Trading Strategy
- **VWAP-Based Execution**: Uses Volume-Weighted Average Price for all trades with hyper-liquidity assumption
- **XGBoost Prediction**: ML-powered price prediction using VWAP-based technical indicators
- **4-Signal Ensemble**: Combines XGBoost predictions with MACD, RSI, and Bollinger Bands
- **Pyramid Trading**: Adds to winning positions when price rises 5+ basis points
- **Variable Stop-Loss**: Adaptive 8-20 basis point stop-loss that scales with gains

### Risk Management
- **Real-time Stop Loss**: Automatic position exit when losses exceed variable thresholds
- **End-of-Day Logic**: Forced liquidation at 3:55 PM, cut losses at 3:50 PM
- **Dynamic Capital Allocation**: 6% for tech stocks, 9% for non-tech stocks
- **Transaction Cost Modeling**: Realistic Alpaca fee structure ($0.001/share + regulatory fees)

### Rolling Backtest System
- **Weekly Retraining**: 12-week rolling window with fresh API data each week
- **Realistic Simulation**: Simulates Jan 5 - Sep 19, 2025 with weekly model updates
- **Per-Ticker P&L Tracking**: Individual performance metrics for each stock
- **Comprehensive Reporting**: Weekly summaries with cumulative statistics

## Performance Features

### Real-Time Monitoring
- Minute-by-minute trading activity with consolidated output
- Portfolio value tracking with return percentages
- Trade execution details (shares, pyramid levels, P&L)
- Prediction accuracy and technical signal alignment

### Weekly Reporting
- Individual ticker performance and P&L
- Cumulative portfolio statistics
- Model training summaries and accuracy metrics
- Capital utilization and drawdown analysis

## Installation

### Prerequisites
- Python 3.8+
- Polygon.io API key (free tier available)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd actual_lab4

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your Polygon.io API key
```

## Configuration

### Environment Variables (.env)
```bash
# Required
POLYGON_API_KEY=your_polygon_api_key_here

# Trading Parameters are currently hardcoded. TODO: Update later
# INITIAL_CAPITAL=500000
# TECH_ALLOCATION_PCT=6/100
# NON_TECH_ALLOCATION_PCT=9/100
# BASE_STOP_LOSS_BPS=8
# MAX_STOP_LOSS_BPS=20
# PREDICTION_THRESHOLD_BPS=2
```

### Default Ticker List
The system trades the following stocks, subject to change:

- **Tech**: AAPL, META, NVDA, PFE, PLTR, UBER, NFLX
- **Non-Tech**: BAC, V

*Note: MCD and PEP were removed due to intraday volatility issues*

## Usage

### Basic Rolling Backtest
```python
#!/usr/bin/env python3
from polygon_xgboost_trading_v6 import PolygonXGBoostHFTTrader
from datetime import datetime, timedelta

# Configuration
API_KEY = "your_polygon_api_key"
TICKERS = ["AAPL", "META", "NVDA", "BAC", "V", "NFLX", "PFE", "PLTR", "UBER"]
INITIAL_CAPITAL = 50000

# Initialize trader
trader = PolygonXGBoostHFTTrader(
    api_key=API_KEY,
    tickers=TICKERS,
    initial_capital=INITIAL_CAPITAL
)

# Run rolling weekly backtest
results = trader.run_rolling_weekly_backtest(
    start_date=datetime(2025, 1, 5),  # First test week
    end_date=datetime(2025, 9, 19),   # Last test week
    training_weeks=12,
    validation_split=0.25
)

print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
print(f"Total Return: {results['total_return_pct']:+.2f}%")
```

### Advanced Configuration
```python
# Custom allocations
allocations = {
    "AAPL": 0.06,    # 6% for tech stocks
    "META": 0.06,
    "NVDA": 0.06,
    "BAC": 0.09,     # 9% for non-tech stocks
    "V": 0.09
}

# Custom constraints
from polygon_xgboost_trading_v6 import TradingConstraints
constraints = TradingConstraints()
constraints.BASE_STOP_LOSS_BPS = 10  # More conservative stop-loss

trader = PolygonXGBoostHFTTrader(
    api_key=API_KEY,
    tickers=list(allocations.keys()),
    initial_capital=100000,
    allocations=allocations,
    constraints=constraints
)
```

## Trading Logic

### Signal Generation
1. **XGBoost Prediction**: Predicts next minute VWAP using 20+ technical features
2. **Technical Indicators**: MACD crossover, RSI levels (30/70), Bollinger Band position
3. **Ensemble Decision**: Majority vote system with prediction confidence weighting
4. **Execution**: Buy/Sell/Pyramid decisions based on 4-signal consensus

### Position Management
- **Initial Buy**: When signals align for bullish outlook
- **Pyramid Add**: When position profitable and signals remain bullish
- **Stop Loss**: When losses exceed variable threshold (8-20 basis points)
- **End-of-Day**: Automatic liquidation at market close

### Risk Controls
- **Prediction Threshold**: Only trade when predictions ≥ ±2 basis points
- **Capital Limits**: Maximum 90% capital utilization
- **Position Sizing**: Dynamic allocation based on stock category
- **Time Limits**: No new positions and cut losses after 3:50PM, force sell at 3:55PM

## Output Structure

```
run_YYYYMMDD_HHMMSS/
├── weekly_cumulative_stats.txt          # Overall performance summary
├── rolling_backtest_config.json         # Configuration snapshot
├── historical_data_TICKER.csv           # Raw market data per ticker
├── week_YYYYMMDD/
│   └── trading_results.csv              # Detailed minute-by-minute results
└── ...
```

### Key Output Files
- **weekly_cumulative_stats.txt**: Week-by-week P&L, portfolio value, return percentages
- **trading_results.csv**: Complete trade history with signals, predictions, and P&L
- **rolling_backtest_config.json**: System configuration and parameters

## Technical Architecture

### Core Components
```
PolygonXGBoostHFTTrader
├── PolygonDataManager     # API data fetching with caching
├── TechnicalAnalysis      # MACD, RSI, Bollinger Bands calculation
├── XGBoostHFTModel       # ML prediction with GridSearchCV optimization
├── EnsembleDecision      # 4-signal decision making logic
└── PositionManager       # Capital allocation, risk management, P&L tracking
```

### Data Flow
```
Polygon.io API → Historical Data → Technical Features → XGBoost Model → Predictions
                                         ↓
                              Technical Indicators → Ensemble Decision → Trade Execution
```

## Risk Warnings

**This is educational/research software. As such, important considerations must be noted:**

- **Paper Trading Only**: This system is designed for backtesting and analysis
- **Market Risk**: Past performance does not guarantee future results
- **API Limits**: Polygon.io has rate limits and data delays
- **Transaction Costs**: Real trading includes slippage, spreads, and market impact
- **Regulatory**: Ensure compliance with local trading regulations

## Testing and Validation

### Backtesting Results (Jan-Sep 2025)
- **Training Period**: 12 weeks rolling window with weekly retraining
- **Test Period**: 37 weeks of live simulation
- **Data Source**: Polygon.io 1-minute VWAP data
- **Validation**: 25% of training data held out for model validation

### Performance Metrics
- Total return percentage
- Per-ticker P&L tracking
- Win/loss ratios
- Maximum drawdown
- Sharpe ratio calculation
- Capital utilization efficiency

## Development

### Code Structure
- **Main System**: `polygon_xgboost_trading_v6.py` (1400+ lines)
- **Clean Architecture**: Object-oriented design with clear separation of concerns
- **Error Handling**: Comprehensive exception handling and data validation
- **Logging**: Detailed trading activity and progress tracking

### Key Classes
- `TradingConstraints`: Fee structure and trading limits
- `PolygonDataManager`: Data fetching and caching
- `TechnicalAnalysis`: Indicator calculation
- `XGBoostHFTModel`: ML model with hyperparameter optimization
- `EnsembleDecision`: Signal aggregation and decision logic
- `PositionManager`: Trade execution and risk management

## Dependencies

See `requirements.txt` for complete list:
- pandas, numpy (data manipulation)
- xgboost, scikit-learn (machine learning)
- polygon-api-client (market data)
- tqdm (progress tracking)

## License

This project is for educational and research purposes. Please ensure compliance with your local trading regulations and use appropriate risk management when adapting for live trading.

## Contributing

This is a research project. When adapting for production use:
1. Implement proper error handling for live market conditions
2. Add real-time data validation
3. Include comprehensive logging and monitoring
4. Test thoroughly in paper trading environment
5. Ensure regulatory compliance

---

**Disclaimer**: This software is for educational purposes only. Trading involves substantial risk and is not suitable for all investors. Always consult with qualified financial professionals before making trading decisions.
