#!/usr/bin/env python3
"""
Polygon.io XGBoost HFT Trading System - Final Version
====================================================

Complete trading system with:
- XGBRegressor for price prediction (1BP threshold)
- 4-signal ensemble system (3/4 required)
- Variable stop-loss (4-12BP from max)
- Pyramid trading (50% sizing, 2BP + 2/3 indicators)
- Capital management (80% allocation limit)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, List, Tuple
from polygon import RESTClient
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class AlpacaConstraints:
    """Trading platform constraints and fees"""
    COMMISSION_PER_SHARE = 0.001
    FINRA_TAF_FEE = 0.000166
    SEC_FEE = 0.0000278
    
    MIN_FRACTIONAL_DOLLARS = 1.00
    MIN_FRACTIONAL_SHARES = 0.001
    MAX_SHARES_PER_ORDER = 10000
    
    # HFT parameters
    BASE_STOP_LOSS_BPS = 4  # Starting stop loss
    MAX_STOP_LOSS_BPS = 12  # Maximum stop loss
    STOP_LOSS_SCALE = 0.08  # BP increase per BP gained


class PolygonDataManager:
    """Manages Polygon.io data fetching and caching"""
    
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)
        self.data_cache = {}
        
    def fetch_ticker_data(self, ticker: str, start_date: datetime, end_date: datetime, 
                         interval: str = "1", timespan: str = "minute") -> pd.DataFrame:
        """Fetch and cache ticker data"""
        cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            bars = self.client.get_aggs(
                ticker=ticker,
                multiplier=int(interval),
                timespan=timespan,
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=50000
            )
            
            df = pd.DataFrame(bars)
            if df.empty:
                return df
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Keep only regular trading hours
            df = df.set_index('timestamp').between_time("09:30", "16:00").reset_index()
            
            self.data_cache[cache_key] = df
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()


class TechnicalIndicators:
    """Calculate technical indicators for ensemble signals"""
    
    def __init__(self):
        self.latest_values = {}
        
    def calculate_all_indicators(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # MACD
        df = self._add_macd(df, ticker)
        
        # RSI
        df = self._add_rsi(df, ticker)
        
        # Bollinger Bands
        df = self._add_bollinger_bands(df, ticker)
        
        # Generate individual signals
        df = self._generate_signals(df, ticker)
        
        return df
    
    def _add_macd(self, df: pd.DataFrame, ticker: str, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df[f'{ticker}_macd'] = ema_fast - ema_slow
        df[f'{ticker}_macd_signal'] = df[f'{ticker}_macd'].ewm(span=signal, adjust=False).mean()
        df[f'{ticker}_macd_histogram'] = df[f'{ticker}_macd'] - df[f'{ticker}_macd_signal']
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame, ticker: str, window: int = 14) -> pd.DataFrame:
        """Add RSI indicator"""
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame, ticker: str, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        
        df[f'{ticker}_bb_upper'] = rolling_mean + (rolling_std * num_std)
        df[f'{ticker}_bb_middle'] = rolling_mean
        df[f'{ticker}_bb_lower'] = rolling_mean - (rolling_std * num_std)
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Generate individual signals from each indicator"""
        # MACD Signal (avoid look-ahead bias)
        macd_cross_up = (df[f'{ticker}_macd'] > df[f'{ticker}_macd_signal']) & \
                       (df[f'{ticker}_macd'].shift(1) <= df[f'{ticker}_macd_signal'].shift(1))
        macd_cross_down = (df[f'{ticker}_macd'] < df[f'{ticker}_macd_signal']) & \
                         (df[f'{ticker}_macd'].shift(1) >= df[f'{ticker}_macd_signal'].shift(1))
        
        df[f'{ticker}_macd_signal_label'] = 0  # Hold
        df.loc[macd_cross_up, f'{ticker}_macd_signal_label'] = 1   # Buy
        df.loc[macd_cross_down, f'{ticker}_macd_signal_label'] = -1  # Sell
        
        # RSI Signal
        df[f'{ticker}_rsi_signal_label'] = 0  # Hold
        df.loc[df[f'{ticker}_rsi'] < 30, f'{ticker}_rsi_signal_label'] = 1   # Oversold = Buy
        df.loc[df[f'{ticker}_rsi'] > 70, f'{ticker}_rsi_signal_label'] = -1  # Overbought = Sell
        
        # Bollinger Bands Signal
        df[f'{ticker}_bb_signal_label'] = 0  # Hold
        df.loc[df['close'] <= df[f'{ticker}_bb_lower'], f'{ticker}_bb_signal_label'] = 1   # Buy
        df.loc[df['close'] >= df[f'{ticker}_bb_upper'], f'{ticker}_bb_signal_label'] = -1  # Sell
        
        return df
    
    def get_current_signals(self, df: pd.DataFrame, ticker: str) -> Dict[str, int]:
        """Get latest signal values for ensemble"""
        if df.empty:
            return {'macd': 0, 'rsi': 0, 'bollinger': 0}
        
        return {
            'macd': int(df[f'{ticker}_macd_signal_label'].iloc[-1]),
            'rsi': int(df[f'{ticker}_rsi_signal_label'].iloc[-1]),
            'bollinger': int(df[f'{ticker}_bb_signal_label'].iloc[-1])
        }


class PricePredictionModel:
    """XGBRegressor for price prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.is_trained = False
        self.min_training_samples = 200
        
    def prepare_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare features for price prediction"""
        df = df.copy()
        feature_cols = []
        
        # Price features (avoid look-ahead bias)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_momentum_3'] = df['close'] - df['close'].shift(3)
        df['price_momentum_5'] = df['close'] - df['close'].shift(5)
        feature_cols.extend(['returns', 'log_returns', 'price_momentum_3', 'price_momentum_5'])
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_momentum'] = df['volume'] - df['volume'].shift(3)
        feature_cols.extend(['volume_ratio', 'volume_momentum'])
        
        # Technical indicators
        tech_cols = [col for col in df.columns if ticker in col and any(
            ind in col for ind in ['macd', 'rsi', 'bb_']
        )]
        feature_cols.extend(tech_cols)
        
        # OHLC features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        feature_cols.extend(['high_low_ratio', 'close_position'])
        
        # Lag features (properly lagged to avoid look-ahead)
        for lag in [1, 2, 3]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            feature_cols.extend([f'close_lag_{lag}', f'volume_lag_{lag}'])
        
        self.feature_cols = feature_cols
        return df
    
    def train_model(self, df: pd.DataFrame) -> bool:
        """Train XGBoost regressor with GridSearchCV"""
        df_clean = df.dropna()
        
        if len(df_clean) < self.min_training_samples:
            print(f"Insufficient data for training: {len(df_clean)} rows")
            return False
        
        # Prepare target (next period's price)
        X = df_clean[self.feature_cols].values
        y = df_clean['close'].shift(-1).dropna().values  # Predict next price
        X = X[:-1]  # Remove last row to match y length
        
        if len(X) != len(y):
            print("Feature/target length mismatch")
            return False
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Grid search parameters
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        # Time series cross validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        try:
            xgb = XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                tree_method='hist',
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                xgb, param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            # Validate
            val_pred = self.model.predict(X_val)
            mse = mean_squared_error(y_val, val_pred)
            
            self.is_trained = True
            print(f"Model trained successfully. Validation MSE: {mse:.4f}")
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def predict_next_price(self, df: pd.DataFrame) -> float:
        """Predict next period's price"""
        if not self.is_trained or df.empty:
            return 0.0
        
        try:
            # Get latest features
            latest_features = df[self.feature_cols].iloc[-1:].values
            
            if np.isnan(latest_features).any():
                return 0.0
            
            predicted_price = self.model.predict(latest_features)[0]
            return float(predicted_price)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0


class PositionManager:
    """Manages capital, positions, and trade execution"""
    
    def __init__(self, tickers: List[str], initial_capital: float):
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {ticker: 0 for ticker in tickers}
        self.buy_prices = {ticker: 0.0 for ticker in tickers}
        self.max_prices = {ticker: 0.0 for ticker in tickers}  # Track max while holding
        self.allocations = {}
        self.constraints = AlpacaConstraints()
        self.pyramid_history = {ticker: 0 for ticker in tickers}  # Count pyramids
        
    def set_allocations(self, allocations: Optional[Dict[str, float]] = None):
        """Set and validate capital allocations"""
        if allocations:
            # Ensure total allocation doesn't exceed 80% of capital
            total_allocated = sum(allocations.values())
            if total_allocated > 0.8 * self.initial_capital:
                scale_factor = (0.8 * self.initial_capital) / total_allocated
                allocations = {k: v * scale_factor for k, v in allocations.items()}
                print(f"Scaled down allocations to maintain 80% limit")
            
            self.allocations = allocations
        else:
            # Equal allocation within 80% limit
            allocation_per_ticker = (0.8 * self.initial_capital) / len(self.tickers)
            self.allocations = {ticker: allocation_per_ticker for ticker in self.tickers}
        
        print(f"Allocations set: {self.allocations}")
    
    def calculate_variable_stop_loss(self, ticker: str, current_price: float) -> float:
        """Calculate variable stop loss threshold"""
        if self.positions[ticker] == 0:
            return 0
        
        # BP gained from buy price
        bp_gain = ((current_price - self.buy_prices[ticker]) / self.buy_prices[ticker]) * 10000
        
        # Variable threshold: 4BP + 0.08BP per BP gained, capped at 12BP
        threshold = self.constraints.BASE_STOP_LOSS_BPS + (bp_gain * self.constraints.STOP_LOSS_SCALE)
        threshold = min(threshold, self.constraints.MAX_STOP_LOSS_BPS)
        
        return max(threshold, self.constraints.BASE_STOP_LOSS_BPS)  # Never below 4BP
    
    def check_safety_net_sell(self, ticker: str, current_price: float) -> bool:
        """Check if safety net auto-sell should trigger"""
        if self.positions[ticker] == 0:
            return False
        
        # Update max price
        if current_price > self.max_prices[ticker]:
            self.max_prices[ticker] = current_price
        
        # Calculate drop from max
        drop_from_max = ((self.max_prices[ticker] - current_price) / self.max_prices[ticker]) * 10000
        
        # Get variable threshold
        threshold = self.calculate_variable_stop_loss(ticker, current_price)
        
        if drop_from_max >= threshold:
            print(f"🛑 Safety net triggered for {ticker}: {drop_from_max:.1f}BP drop (threshold: {threshold:.1f}BP)")
            return True
        
        return False
    
    def calculate_position_size(self, ticker: str, current_price: float, is_pyramid: bool = False) -> int:
        """Calculate shares to buy"""
        if is_pyramid:
            # Pyramid: 50% of original allocation
            max_spend = self.allocations[ticker] * 0.5
        else:
            # Initial: full allocation
            max_spend = self.allocations[ticker]
        
        # Don't exceed available capital
        max_spend = min(max_spend, self.current_capital * 0.9)  # Keep 10% buffer
        
        shares = int(max_spend / current_price)
        
        # Apply constraints
        if shares < self.constraints.MIN_FRACTIONAL_SHARES:
            return 0
        if shares > self.constraints.MAX_SHARES_PER_ORDER:
            shares = self.constraints.MAX_SHARES_PER_ORDER
        
        return shares
    
    def calculate_transaction_cost(self, shares: int, is_sell: bool = False) -> float:
        """Calculate transaction costs"""
        commission = self.constraints.COMMISSION_PER_SHARE * shares
        
        if is_sell:
            finra_taf = self.constraints.FINRA_TAF_FEE * shares
            sec_fee = self.constraints.SEC_FEE * shares
            return commission + finra_taf + sec_fee
        
        return commission
    
    def execute_buy(self, ticker: str, current_price: float, is_pyramid: bool = False) -> bool:
        """Execute buy order"""
        shares = self.calculate_position_size(ticker, current_price, is_pyramid)
        
        if shares == 0:
            return False
        
        cost = shares * current_price + self.calculate_transaction_cost(shares)
        
        if cost > self.current_capital:
            return False
        
        # Execute trade
        self.current_capital -= cost
        
        # Update position tracking
        if self.positions[ticker] == 0:
            # New position
            self.buy_prices[ticker] = current_price
            self.max_prices[ticker] = current_price
            self.pyramid_history[ticker] = 0
        else:
            # Pyramid: update average buy price
            total_value = (self.positions[ticker] * self.buy_prices[ticker]) + (shares * current_price)
            total_shares = self.positions[ticker] + shares
            self.buy_prices[ticker] = total_value / total_shares
            self.pyramid_history[ticker] += 1
        
        self.positions[ticker] += shares
        
        action = f"Pyramid {self.pyramid_history[ticker]}" if is_pyramid else "Initial buy"
        print(f"{action} {ticker}: {shares} shares @ ${current_price:.2f} (Total: {self.positions[ticker]})")
        
        return True
    
    def execute_sell(self, ticker: str, current_price: float) -> bool:
        """Execute sell order (sell all shares)"""
        shares = self.positions[ticker]
        
        if shares == 0:
            return False
        
        # Calculate earnings
        earnings = shares * current_price - self.calculate_transaction_cost(shares, is_sell=True)
        
        # Calculate P&L
        total_cost = shares * self.buy_prices[ticker]
        profit_loss = earnings - total_cost
        pnl_pct = (profit_loss / total_cost) * 100
        
        # Execute trade
        self.current_capital += earnings
        
        # Reset position tracking
        self.positions[ticker] = 0
        self.buy_prices[ticker] = 0.0
        self.max_prices[ticker] = 0.0
        self.pyramid_history[ticker] = 0
        
        emoji = "PROFIT" if profit_loss > 0 else "LOSS"
        print(f"{emoji} Sold all {shares} shares of {ticker} @ ${current_price:.2f}")
        print(f"   P&L: ${profit_loss:.2f} ({pnl_pct:+.2f}%)")
        
        return True
    
    def get_portfolio_status(self, current_prices: Dict[str, float]) -> Dict:
        """Get portfolio status"""
        stock_value = sum(
            self.positions[ticker] * current_prices.get(ticker, 0)
            for ticker in self.tickers
        )
        
        portfolio_value = self.current_capital + stock_value
        
        return {
            'portfolio_value': portfolio_value,
            'current_capital': self.current_capital,
            'stock_value': stock_value,
            'positions': self.positions.copy(),
            'return_pct': ((portfolio_value - self.initial_capital) / self.initial_capital) * 100,
            'capital_usage': 1 - (self.current_capital / self.initial_capital)
        }


class EnsembleDecisionMaker:
    """Makes trading decisions using 4-signal ensemble"""
    
    def __init__(self):
        self.signals_history = []
        
    def make_decision(self, prediction_signal: int, technical_signals: Dict[str, int], 
                     holding_position: bool) -> str:
        """
        Make trading decision using ensemble rules:
        - Initial position: 3/4 signals = Buy
        - When holding: 3/4 sell signals = Sell, strong bullish = Pyramid
        """
        # Collect all signals
        all_signals = [
            prediction_signal,
            technical_signals['macd'],
            technical_signals['rsi'],
            technical_signals['bollinger']
        ]
        
        buy_count = sum(1 for s in all_signals if s == 1)
        sell_count = sum(1 for s in all_signals if s == -1)
        
        if not holding_position:
            # Need 3/4 buy signals for initial position
            if buy_count >= 3:
                return "Buy"
        else:
            # When holding position
            if sell_count >= 3:
                return "Sell"
            elif prediction_signal == 1 and buy_count >= 2:
                # Pyramid conditions: prediction bullish + at least 1 other
                return "Pyramid"
        
        return "Hold"
    
    def can_pyramid(self, predicted_bp: float, technical_signals: Dict[str, int]) -> bool:
        """Check if pyramid conditions are met"""
        # Need +2BP prediction AND 2/3 other indicators bullish
        if predicted_bp < 2.0:
            return False
        
        other_bullish = sum(1 for signal in [
            technical_signals['macd'],
            technical_signals['rsi'], 
            technical_signals['bollinger']
        ] if signal == 1)
        
        return other_bullish >= 2


class PolygonXGBoostTrader:
    """Main trading system orchestrator"""
    
    def __init__(self, api_key: str, tickers: List[str], initial_capital: float,
                 total_weeks: int = 12, train_weeks: int = 10, test_weeks: int = 2):
        
        self.data_manager = PolygonDataManager(api_key)
        self.technical_indicators = TechnicalIndicators()
        self.position_manager = PositionManager(tickers, initial_capital)
        self.ensemble_decision = EnsembleDecisionMaker()
        
        self.tickers = tickers
        self.total_weeks = total_weeks
        self.train_weeks = train_weeks
        self.test_weeks = test_weeks
        
        # Models per ticker
        self.models = {ticker: PricePredictionModel() for ticker in tickers}
        
        # Performance tracking
        self.performance_history = []
        
    def initialize_system(self, allocations: Optional[Dict[str, float]] = None):
        """Initialize the trading system"""
        self.position_manager.set_allocations(allocations)
        print(f"Trading System Initialized")
        print(f"   Capital: ${self.position_manager.initial_capital:,.2f}")
        print(f"   Tickers: {', '.join(self.tickers)}")
    
    def fetch_and_prepare_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch and prepare data for all tickers"""
        prepared_data = {}
        
        for ticker in self.tickers:
            print(f"Processing {ticker}...")
            
            # Fetch raw data (1-minute bars)
            df = self.data_manager.fetch_ticker_data(ticker, start_date, end_date, interval="1")
            
            if df.empty:
                print(f"   No data available")
                continue
            
            # Add technical indicators
            df = self.technical_indicators.calculate_all_indicators(df, ticker)
            
            # Prepare features for ML
            df = self.models[ticker].prepare_features(df, ticker)
            
            prepared_data[ticker] = df
            print(f"   {len(df)} data points processed")
        
        return prepared_data
    
    def train_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Train prediction models for all tickers"""
        results = {}
        
        for ticker, df in data.items():
            print(f"Training {ticker} model...")
            success = self.models[ticker].train_model(df)
            results[ticker] = success
            
            if success:
                print(f"   Model trained successfully")
            else:
                print(f"   Training failed")
        
        return results
    
    def execute_trading_cycle(self, current_data: Dict[str, pd.DataFrame]) -> Dict:
        """Execute one complete trading cycle"""
        cycle_results = {}
        current_prices = {}
        
        # Get current prices
        for ticker in self.tickers:
            if ticker in current_data and len(current_data[ticker]) > 0:
                current_prices[ticker] = current_data[ticker]['close'].iloc[-1]
        
        # Execute trades for each ticker
        for ticker in self.tickers:
            if ticker not in current_data or not self.models[ticker].is_trained:
                continue
            
            df = current_data[ticker]
            current_price = current_prices[ticker]
            
            # 1. Check safety net first (always overrides)
            if self.position_manager.check_safety_net_sell(ticker, current_price):
                self.position_manager.execute_sell(ticker, current_price)
                cycle_results[ticker] = 'SafetyNetSell'
                continue
            
            # 2. Get prediction signal
            predicted_price = self.models[ticker].predict_next_price(df)
            predicted_bp = 0
            prediction_signal = 0
            
            if predicted_price > 0:
                predicted_bp = ((predicted_price - current_price) / current_price) * 10000
                if predicted_bp >= 1.0:
                    prediction_signal = 1  # Buy
                elif predicted_bp <= -1.0:
                    prediction_signal = -1  # Sell
            
            # 3. Get technical signals
            technical_signals = self.technical_indicators.get_current_signals(df, ticker)
            
            # 4. Make ensemble decision
            holding_position = self.position_manager.positions[ticker] > 0
            decision = self.ensemble_decision.make_decision(
                prediction_signal, technical_signals, holding_position
            )
            
            # 5. Execute decision
            if decision == "Buy":
                if self.position_manager.execute_buy(ticker, current_price):
                    cycle_results[ticker] = 'Buy'
                else:
                    cycle_results[ticker] = 'BuyFailed'
                    
            elif decision == "Sell":
                if self.position_manager.execute_sell(ticker, current_price):
                    cycle_results[ticker] = 'Sell'
                else:
                    cycle_results[ticker] = 'SellFailed'
                    
            elif decision == "Pyramid":
                # Additional check for pyramid conditions
                if self.ensemble_decision.can_pyramid(predicted_bp, technical_signals):
                    if self.position_manager.execute_buy(ticker, current_price, is_pyramid=True):
                        cycle_results[ticker] = 'Pyramid'
                    else:
                        cycle_results[ticker] = 'PyramidFailed'
                else:
                    cycle_results[ticker] = 'Hold'
            else:
                cycle_results[ticker] = 'Hold'
        
        # Get portfolio status
        portfolio_status = self.position_manager.get_portfolio_status(current_prices)
        cycle_results['portfolio'] = portfolio_status
        
        return cycle_results
    
    def run_weekly_update(self) -> Dict:
        """Weekly update: fetch new data, retrain, execute trades"""
        print("\n" + "="*60)
        print("WEEKLY UPDATE")
        print("="*60)
        
        # Fetch new 12-week window
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=self.total_weeks)
        
        # Prepare data
        new_data = self.fetch_and_prepare_data(start_date, end_date)
        
        # Retrain models
        print("\nRetraining models...")
        training_results = self.train_models(new_data)
        
        # Execute trading cycle
        print("\n💼 Executing trades...")
        trading_results = self.execute_trading_cycle(new_data)
        
        # Store performance
        self.performance_history.append({
            'date': datetime.now(),
            'training_results': training_results,
            'trading_results': trading_results,
            'portfolio_value': trading_results['portfolio']['portfolio_value']
        })
        
        return trading_results
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        if not self.performance_history:
            return {}
        
        values = [record['portfolio_value'] for record in self.performance_history]
        
        return {
            'current_value': values[-1],
            'initial_value': self.position_manager.initial_capital,
            'total_return': ((values[-1] - self.position_manager.initial_capital) / 
                           self.position_manager.initial_capital) * 100,
            'max_value': max(values),
            'min_value': min(values),
            'volatility': np.std(values) if len(values) > 1 else 0
        }


def main():
    """Example usage"""
    # Configuration
    # TODO: Change this!!!
    API_KEY = "ySVAijwYrecApsiZErX3ETrPTr46ygXC"
    INITIAL_CAPITAL = 500000
    TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT"]
    
    # Custom allocations (optional)
    ALLOCATIONS = {
        "AAPL": 12000,
        "TSLA": 15000,
        "NVDA": 10000,
        "MSFT": 8000
    }
    
    # Initialize trader
    trader = PolygonXGBoostTrader(
        api_key=API_KEY,
        tickers=TICKERS,
        initial_capital=INITIAL_CAPITAL
    )
    
    # Initialize system
    trader.initialize_system(ALLOCATIONS)
    
    # Initial training
    print("\nINITIAL SETUP")
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=12)
    
    initial_data = trader.fetch_and_prepare_data(start_date, end_date)
    training_results = trader.train_models(initial_data)
    
    # Execute initial trades
    trading_results = trader.execute_trading_cycle(initial_data)
    
    print("\nInitial Portfolio Status:")
    portfolio = trading_results['portfolio']
    print(f"   Value: ${portfolio['portfolio_value']:,.2f}")
    print(f"   Return: {portfolio['return_pct']:+.2f}%")
    print(f"   Positions: {portfolio['positions']}")
    
    # Simulate weekly updates
    for week in range(4):
        results = trader.run_weekly_update()
        
        print(f"\nWeek {week + 1} Results:")
        portfolio = results['portfolio']
        print(f"   Value: ${portfolio['portfolio_value']:,.2f}")
        print(f"   Return: {portfolio['return_pct']:+.2f}%")
        
        # Show trade actions
        actions = {k: v for k, v in results.items() if k != 'portfolio'}
        print(f"   Actions: {actions}")
    
    # Final summary
    summary = trader.get_performance_summary()
    print("\n" + "="*60)
    print("FINAL PERFORMANCE")
    print("="*60)
    print(f"Total Return: {summary['total_return']:+.2f}%")
    print(f"Max Value: ${summary['max_value']:,.2f}")
    print(f"Min Value: ${summary['min_value']:,.2f}")


if __name__ == "__main__":
    main()