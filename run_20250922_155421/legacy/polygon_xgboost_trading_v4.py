#!/usr/bin/env python3
"""
Polygon.io XGBoost HFT Trading System - Fixed Version
====================================================

Fixes:
- Forward fill for missing data (backfill only at market open)
- Trading restricted to 10:00-16:00
- Use VWAP and Volume from Polygon.io
- Almost never dropna() - handle NaN properly
- Proper market hours handling
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
    BASE_STOP_LOSS_BPS = 4
    MAX_STOP_LOSS_BPS = 12
    STOP_LOSS_SCALE = 0.08


class PolygonDataManager:
    """Manages Polygon.io data fetching with proper market hours"""
    
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)
        self.data_cache = {}
        
    def fetch_ticker_data(self, ticker: str, start_date: datetime, end_date: datetime, 
                         interval: str = "1", timespan: str = "minute") -> pd.DataFrame:
        """Fetch ticker data with VWAP and Volume"""
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
            
            # Keep ALL trading hours first (9:30-16:00) for data completeness
            df = df.set_index('timestamp').between_time("09:30", "16:00").reset_index()
            
            # Ensure we have VWAP and volume
            if 'vwap' not in df.columns:
                df['vwap'] = df['close']  # Fallback to close if no VWAP
            if 'volume' not in df.columns:
                df['volume'] = 0  # Fallback
            
            # Create complete time index for the trading day
            df = self._fill_missing_timestamps(df)
            
            self.data_cache[cache_key] = df
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _fill_missing_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing timestamps and handle gaps properly"""
        if df.empty:
            return df
        
        # Set timestamp as index
        df = df.set_index('timestamp').sort_index()
        
        # Create complete minute-by-minute index for trading hours
        start_date = df.index.min().normalize()
        end_date = df.index.max().normalize() + pd.Timedelta(days=1)
        
        # Generate all trading minutes
        all_minutes = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday-Friday only
                day_start = current_date + pd.Timedelta(hours=9, minutes=30)
                day_end = current_date + pd.Timedelta(hours=16)
                day_minutes = pd.date_range(day_start, day_end, freq='1min')[:-1]  # Exclude 16:00
                all_minutes.extend(day_minutes)
            current_date += pd.Timedelta(days=1)
        
        if not all_minutes:
            return df.reset_index()
        
        complete_index = pd.DatetimeIndex(all_minutes)
        complete_index = complete_index[complete_index.isin(df.index.union(complete_index))]
        
        # Reindex with complete trading minutes
        df_complete = df.reindex(complete_index)
        
        # Handle missing data:
        # 1. At market open (9:30): backfill if needed
        # 2. During trading: forward fill (market hasn't moved)
        
        # First, backfill only the very first values of each day
        df_complete = df_complete.groupby(df_complete.index.date).apply(
            lambda group: group.bfill(limit=1)  # Only backfill first minute of day
        ).droplevel(0)
        
        # Then forward fill everything else
        df_complete = df_complete.ffill()
        
        # For any remaining NaN (shouldn't happen), use interpolation
        df_complete = df_complete.interpolate(method='linear')
        
        return df_complete.reset_index()
    
    def is_trading_time(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within trading hours (10:00-16:00)"""
        time_of_day = timestamp.time()
        return (time_of_day >= pd.Timestamp('10:00').time() and 
                time_of_day < pd.Timestamp('16:00').time() and
                timestamp.weekday() < 5)


class TechnicalIndicators:
    """Calculate technical indicators with proper NaN handling"""
    
    def __init__(self):
        self.latest_values = {}
        
    def calculate_all_indicators(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate all technical indicators without dropping data"""
        df = df.copy()
        
        # Only calculate if we have enough data
        if len(df) < 50:  # Need minimum data for indicators
            return df
        
        # MACD
        df = self._add_macd(df, ticker)
        
        # RSI
        df = self._add_rsi(df, ticker)
        
        # Bollinger Bands
        df = self._add_bollinger_bands(df, ticker)
        
        # Generate signals (handle NaN gracefully)
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
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        df[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame, ticker: str, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        rolling_mean = df['close'].rolling(window=window, min_periods=1).mean()
        rolling_std = df['close'].rolling(window=window, min_periods=1).std()
        
        df[f'{ticker}_bb_upper'] = rolling_mean + (rolling_std * num_std)
        df[f'{ticker}_bb_middle'] = rolling_mean
        df[f'{ticker}_bb_lower'] = rolling_mean - (rolling_std * num_std)
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Generate signals with NaN handling"""
        # Initialize all signals as Hold (0)
        df[f'{ticker}_macd_signal_label'] = 0
        df[f'{ticker}_rsi_signal_label'] = 0
        df[f'{ticker}_bb_signal_label'] = 0
        
        # MACD Signal (only where we have valid data)
        macd_valid = df[f'{ticker}_macd'].notna() & df[f'{ticker}_macd_signal'].notna()
        if macd_valid.any():
            macd_cross_up = (df[f'{ticker}_macd'] > df[f'{ticker}_macd_signal']) & \
                           (df[f'{ticker}_macd'].shift(1) <= df[f'{ticker}_macd_signal'].shift(1))
            macd_cross_down = (df[f'{ticker}_macd'] < df[f'{ticker}_macd_signal']) & \
                             (df[f'{ticker}_macd'].shift(1) >= df[f'{ticker}_macd_signal'].shift(1))
            
            df.loc[macd_cross_up & macd_valid, f'{ticker}_macd_signal_label'] = 1   # Buy
            df.loc[macd_cross_down & macd_valid, f'{ticker}_macd_signal_label'] = -1  # Sell
        
        # RSI Signal (only where valid)
        rsi_valid = df[f'{ticker}_rsi'].notna()
        if rsi_valid.any():
            df.loc[(df[f'{ticker}_rsi'] < 30) & rsi_valid, f'{ticker}_rsi_signal_label'] = 1   # Buy
            df.loc[(df[f'{ticker}_rsi'] > 70) & rsi_valid, f'{ticker}_rsi_signal_label'] = -1  # Sell
        
        # Bollinger Bands Signal (only where valid)
        bb_valid = df[f'{ticker}_bb_upper'].notna() & df[f'{ticker}_bb_lower'].notna()
        if bb_valid.any():
            df.loc[(df['close'] <= df[f'{ticker}_bb_lower']) & bb_valid, f'{ticker}_bb_signal_label'] = 1   # Buy
            df.loc[(df['close'] >= df[f'{ticker}_bb_upper']) & bb_valid, f'{ticker}_bb_signal_label'] = -1  # Sell
        
        return df
    
    def get_current_signals(self, df: pd.DataFrame, ticker: str) -> Dict[str, int]:
        """Get latest signal values for ensemble"""
        if df.empty:
            return {'macd': 0, 'rsi': 0, 'bollinger': 0}
        
        # Get last valid signals
        macd_signal = df[f'{ticker}_macd_signal_label'].iloc[-1] if f'{ticker}_macd_signal_label' in df.columns else 0
        rsi_signal = df[f'{ticker}_rsi_signal_label'].iloc[-1] if f'{ticker}_rsi_signal_label' in df.columns else 0
        bb_signal = df[f'{ticker}_bb_signal_label'].iloc[-1] if f'{ticker}_bb_signal_label' in df.columns else 0
        
        # Handle NaN
        if pd.isna(macd_signal): macd_signal = 0
        if pd.isna(rsi_signal): rsi_signal = 0
        if pd.isna(bb_signal): bb_signal = 0
        
        return {
            'macd': int(macd_signal),
            'rsi': int(rsi_signal),
            'bollinger': int(bb_signal)
        }


class PricePredictionModel:
    """XGBRegressor for price prediction with robust NaN handling"""
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.is_trained = False
        self.min_training_samples = 100  # Reduced minimum
        
    def prepare_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare features with minimal NaN creation"""
        df = df.copy()
        feature_cols = []
        
        # Basic price features (use fillna for safety)
        df['returns'] = df['close'].pct_change().fillna(0)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        feature_cols.extend(['returns', 'log_returns'])
        
        # Volume features using Polygon.io VWAP and Volume
        df['volume_ratio'] = (df['volume'] / df['volume'].rolling(20, min_periods=1).mean()).fillna(1)
        df['vwap_close_ratio'] = (df['close'] / df['vwap']).fillna(1)
        feature_cols.extend(['volume_ratio', 'vwap_close_ratio', 'volume', 'vwap'])
        
        # Technical indicators (already calculated, use as-is)
        tech_cols = [col for col in df.columns if ticker in col and any(
            ind in col for ind in ['macd', 'rsi', 'bb_']
        ) and not col.endswith('_signal_label')]
        
        # Fill NaN in technical indicators with neutral values
        for col in tech_cols:
            if col in df.columns:
                if 'rsi' in col:
                    df[col] = df[col].fillna(50)  # Neutral RSI
                else:
                    df[col] = df[col].fillna(0)   # Neutral for MACD, BB
                feature_cols.append(col)
        
        # OHLC features
        df['high_low_ratio'] = (df['high'] / df['low']).fillna(1)
        df['close_position'] = ((df['close'] - df['low']) / (df['high'] - df['low'])).fillna(0.5)
        feature_cols.extend(['high_low_ratio', 'close_position'])
        
        # Minimal lag features (only 1 and 2)
        for lag in [1, 2]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag).fillna(method='bfill')
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag).fillna(method='bfill')
            feature_cols.extend([f'close_lag_{lag}', f'volume_lag_{lag}'])
        
        self.feature_cols = feature_cols
        return df
    
    def train_model(self, df: pd.DataFrame) -> bool:
        """Train model with robust data handling"""
        if len(df) < self.min_training_samples:
            print(f"Insufficient data for training: {len(df)} rows")
            return False
        
        # Prepare features
        df_features = df.copy()
        
        # Create target (next period's price)
        df_features['target'] = df_features['close'].shift(-1)
        
        # Only use rows where we have target and current data
        valid_rows = df_features['target'].notna() & df_features['close'].notna()
        
        if not valid_rows.any():
            print("No valid training data")
            return False
        
        df_valid = df_features[valid_rows].copy()
        
        # Ensure all features exist and handle any remaining NaN
        missing_features = [col for col in self.feature_cols if col not in df_valid.columns]
        for col in missing_features:
            df_valid[col] = 0  # Add missing features as zeros
        
        # Final NaN check - fill with appropriate values
        for col in self.feature_cols:
            if df_valid[col].isna().any():
                if 'rsi' in col:
                    df_valid[col] = df_valid[col].fillna(50)
                elif 'ratio' in col:
                    df_valid[col] = df_valid[col].fillna(1)
                else:
                    df_valid[col] = df_valid[col].fillna(0)
        
        X = df_valid[self.feature_cols].values
        y = df_valid['target'].values
        
        if len(X) < self.min_training_samples:
            print(f"Insufficient valid data after cleaning: {len(X)} rows")
            return False
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Simplified parameters for faster training
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
        }
        
        try:
            xgb = XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                tree_method='hist',
                n_jobs=-1
            )
            
            # Simple time series split
            tscv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(
                xgb, param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            # Quick validation
            val_pred = self.model.predict(X_val)
            mse = mean_squared_error(y_val, val_pred)
            
            self.is_trained = True
            print(f"Model trained successfully. Validation MSE: {mse:.4f}, Data points: {len(X)}")
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def predict_next_price(self, df: pd.DataFrame) -> float:
        """Predict next price with robust error handling"""
        if not self.is_trained or df.empty:
            return 0.0
        
        try:
            # Get latest row
            latest_row = df.iloc[-1:].copy()
            
            # Ensure all features exist
            missing_features = [col for col in self.feature_cols if col not in latest_row.columns]
            for col in missing_features:
                latest_row[col] = 0
            
            # Handle NaN in features
            for col in self.feature_cols:
                if latest_row[col].isna().any():
                    if 'rsi' in col:
                        latest_row[col] = latest_row[col].fillna(50)
                    elif 'ratio' in col:
                        latest_row[col] = latest_row[col].fillna(1)
                    else:
                        latest_row[col] = latest_row[col].fillna(0)
            
            features = latest_row[self.feature_cols].values
            
            if np.isnan(features).any():
                print("Warning: NaN in prediction features")
                return 0.0
            
            predicted_price = self.model.predict(features)[0]
            return float(predicted_price)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0


class PositionManager:
    """Position management with trading hours restriction"""
    
    def __init__(self, tickers: List[str], initial_capital: float):
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {ticker: 0 for ticker in tickers}
        self.buy_prices = {ticker: 0.0 for ticker in tickers}
        self.max_prices = {ticker: 0.0 for ticker in tickers}
        self.allocations = {}
        self.constraints = AlpacaConstraints()
        self.pyramid_history = {ticker: 0 for ticker in tickers}
        
    def set_allocations(self, allocations: Optional[Dict[str, float]] = None):
        """Set allocations with 80% limit"""
        if allocations:
            total_allocated = sum(allocations.values())
            if total_allocated > 0.8 * self.initial_capital:
                scale_factor = (0.8 * self.initial_capital) / total_allocated
                allocations = {k: v * scale_factor for k, v in allocations.items()}
                print(f"Scaled down allocations to maintain 80% limit")
            
            self.allocations = allocations
        else:
            allocation_per_ticker = (0.8 * self.initial_capital) / len(self.tickers)
            self.allocations = {ticker: allocation_per_ticker for ticker in self.tickers}
    
    def can_trade_now(self, timestamp: pd.Timestamp) -> bool:
        """Check if we can trade at this time (10:00-16:00)"""
        if pd.isna(timestamp):
            return False
        
        time_of_day = timestamp.time()
        is_trading_hours = (time_of_day >= pd.Timestamp('10:00').time() and 
                           time_of_day < pd.Timestamp('16:00').time())
        is_weekday = timestamp.weekday() < 5
        
        return is_trading_hours and is_weekday
    
    def calculate_variable_stop_loss(self, ticker: str, current_price: float) -> float:
        """Calculate variable stop loss"""
        if self.positions[ticker] == 0:
            return 0
        
        bp_gain = ((current_price - self.buy_prices[ticker]) / self.buy_prices[ticker]) * 10000
        threshold = self.constraints.BASE_STOP_LOSS_BPS + (bp_gain * self.constraints.STOP_LOSS_SCALE)
        threshold = min(threshold, self.constraints.MAX_STOP_LOSS_BPS)
        
        return max(threshold, self.constraints.BASE_STOP_LOSS_BPS)
    
    def check_safety_net_sell(self, ticker: str, current_price: float) -> bool:
        """Safety net with variable threshold"""
        if self.positions[ticker] == 0:
            return False
        
        if current_price > self.max_prices[ticker]:
            self.max_prices[ticker] = current_price
        
        drop_from_max = ((self.max_prices[ticker] - current_price) / self.max_prices[ticker]) * 10000
        threshold = self.calculate_variable_stop_loss(ticker, current_price)
        
        if drop_from_max >= threshold:
            print(f"🛑 Safety net triggered for {ticker}: {drop_from_max:.1f}BP drop")
            return True
        
        return False
    
    def calculate_position_size(self, ticker: str, current_price: float, is_pyramid: bool = False) -> int:
        """Calculate position size"""
        if is_pyramid:
            max_spend = self.allocations[ticker] * 0.5
        else:
            max_spend = self.allocations[ticker]
        
        max_spend = min(max_spend, self.current_capital * 0.9)
        shares = int(max_spend / current_price)
        
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
        """Execute buy with transaction costs"""
        shares = self.calculate_position_size(ticker, current_price, is_pyramid)
        
        if shares == 0:
            return False
        
        cost = shares * current_price + self.calculate_transaction_cost(shares)
        
        if cost > self.current_capital:
            return False
        
        self.current_capital -= cost
        
        if self.positions[ticker] == 0:
            self.buy_prices[ticker] = current_price
            self.max_prices[ticker] = current_price
            self.pyramid_history[ticker] = 0
        else:
            # Update average buy price
            total_value = (self.positions[ticker] * self.buy_prices[ticker]) + (shares * current_price)
            total_shares = self.positions[ticker] + shares
            self.buy_prices[ticker] = total_value / total_shares
            self.pyramid_history[ticker] += 1
        
        self.positions[ticker] += shares
        
        action = f"Pyramid {self.pyramid_history[ticker]}" if is_pyramid else "Buy"
        print(f"{action} {ticker}: {shares} shares @ ${current_price:.2f}")
        
        return True
    
    def execute_sell(self, ticker: str, current_price: float) -> bool:
        """Execute sell with P&L tracking"""
        shares = self.positions[ticker]
        
        if shares == 0:
            return False
        
        earnings = shares * current_price - self.calculate_transaction_cost(shares, is_sell=True)
        total_cost = shares * self.buy_prices[ticker]
        profit_loss = earnings - total_cost
        pnl_pct = (profit_loss / total_cost) * 100
        
        self.current_capital += earnings
        
        # Reset position
        self.positions[ticker] = 0
        self.buy_prices[ticker] = 0.0
        self.max_prices[ticker] = 0.0
        self.pyramid_history[ticker] = 0
        
        emoji = "PROFIT" if profit_loss > 0 else "LOSS"
        print(f"{emoji} Sold {ticker}: {shares} shares @ ${current_price:.2f} | P&L: ${profit_loss:.2f} ({pnl_pct:+.2f}%)")
        
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
    """4-signal ensemble decision maker"""
    
    def make_decision(self, prediction_signal: int, technical_signals: Dict[str, int], 
                     holding_position: bool, predicted_bp: float = 0) -> str:
        """Make ensemble trading decision"""
        all_signals = [
            prediction_signal,
            technical_signals['macd'],
            technical_signals['rsi'],
            technical_signals['bollinger']
        ]
        
        buy_count = sum(1 for s in all_signals if s == 1)
        sell_count = sum(1 for s in all_signals if s == -1)
        
        if not holding_position:
            # Need 3/4 buy signals
            if buy_count >= 3:
                return "Buy"
        else:
            # When holding
            if sell_count >= 3:
                return "Sell"
            elif prediction_signal == 1 and predicted_bp >= 2.0:
                # Pyramid: need +2BP prediction + strong signal
                other_bullish = sum(1 for signal in [
                    technical_signals['macd'],
                    technical_signals['rsi'], 
                    technical_signals['bollinger']
                ] if signal == 1)
                
                if other_bullish >= 2:  # 2/3 other indicators bullish
                    return "Pyramid"
        
        return "Hold"


class PolygonXGBoostTrader:
    """Main trading system with proper market hours"""
    
    def __init__(self, api_key: str, tickers: List[str], initial_capital: float):
        self.data_manager = PolygonDataManager(api_key)
        self.technical_indicators = TechnicalIndicators()
        self.position_manager = PositionManager(tickers, initial_capital)
        self.ensemble_decision = EnsembleDecisionMaker()
        
        self.tickers = tickers
        self.models = {ticker: PricePredictionModel() for ticker in tickers}
        self.performance_history = []
        
    def initialize_system(self, allocations: Optional[Dict[str, float]] = None):
        """Initialize system"""
        self.position_manager.set_allocations(allocations)
        print(f"Trading System Initialized")
        print(f"   Capital: ${self.position_manager.initial_capital:,.2f}")
        print(f"   Trading Hours: 10:00-16:00 EST")
        print(f"   Tickers: {', '.join(self.tickers)}")
    
    def fetch_and_prepare_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch and prepare data with proper filling"""
        prepared_data = {}
        
        for ticker in self.tickers:
            print(f"Processing {ticker}...")
            
            # Fetch with VWAP and Volume
            df = self.data_manager.fetch_ticker_data(ticker, start_date, end_date, interval="1")
            
            if df.empty:
                print(f"   No data available")
                continue
            
            # Add technical indicators
            df = self.technical_indicators.calculate_all_indicators(df, ticker)
            
            # Prepare ML features
            df = self.models[ticker].prepare_features(df, ticker)
            
            prepared_data[ticker] = df
            print(f"   {len(df)} data points | NaN count: {df.isna().sum().sum()}")
        
        return prepared_data
    
    def train_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Train models with robust data handling"""
        results = {}
        
        for ticker, df in data.items():
            print(f"Training {ticker} model...")
            success = self.models[ticker].train_model(df)
            results[ticker] = success
        
        return results
    
    def execute_trading_cycle(self, current_data: Dict[str, pd.DataFrame]) -> Dict:
        """Execute trading with market hours restriction"""
        cycle_results = {}
        current_prices = {}
        
        # Get current timestamp and prices
        current_time = None
        for ticker in self.tickers:
            if ticker in current_data and len(current_data[ticker]) > 0:
                current_prices[ticker] = current_data[ticker]['close'].iloc[-1]
                if current_time is None:
                    current_time = current_data[ticker]['timestamp'].iloc[-1]
        
        # Check if we can trade now
        if current_time is None or not self.position_manager.can_trade_now(current_time):
            print(f"⏰ Outside trading hours: {current_time}")
            return {'portfolio': self.position_manager.get_portfolio_status(current_prices)}
        
        # Execute trades
        for ticker in self.tickers:
            if ticker not in current_data or not self.models[ticker].is_trained:
                continue
            
            df = current_data[ticker]
            current_price = current_prices[ticker]
            
            # 1. Safety net check
            if self.position_manager.check_safety_net_sell(ticker, current_price):
                self.position_manager.execute_sell(ticker, current_price)
                cycle_results[ticker] = 'SafetyNetSell'
                continue
            
            # 2. Get prediction
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
            
            # 4. Make decision
            holding_position = self.position_manager.positions[ticker] > 0
            decision = self.ensemble_decision.make_decision(
                prediction_signal, technical_signals, holding_position, predicted_bp
            )
            
            # 5. Execute
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
                if self.position_manager.execute_buy(ticker, current_price, is_pyramid=True):
                    cycle_results[ticker] = 'Pyramid'
                else:
                    cycle_results[ticker] = 'PyramidFailed'
            else:
                cycle_results[ticker] = 'Hold'
        
        # Portfolio status
        portfolio_status = self.position_manager.get_portfolio_status(current_prices)
        cycle_results['portfolio'] = portfolio_status
        
        return cycle_results
    
    def run_weekly_update(self) -> Dict:
        """Weekly update with comprehensive data handling"""
        print("\n" + "="*60)
        print("WEEKLY UPDATE")
        print("="*60)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=12)
        
        # Fetch and prepare data
        new_data = self.fetch_and_prepare_data(start_date, end_date)
        
        # Train models
        print("\nRetraining models...")
        training_results = self.train_models(new_data)
        
        # Execute trades
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


def main():
    """Example usage with proper API key"""
    # Configuration  
    API_KEY = "ySVAijwYrecApsiZErX3ETrPTr46ygXC"  # Your Polygon.io API key
    INITIAL_CAPITAL = 500000
    TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT"]
    
    # Allocations
    ALLOCATIONS = {
        "AAPL": 120000,
        "TSLA": 150000,
        "NVDA": 100000,
        "MSFT": 80000
    }
    
    # Initialize trader
    trader = PolygonXGBoostTrader(
        api_key=API_KEY,
        tickers=TICKERS,
        initial_capital=INITIAL_CAPITAL
    )
    
    trader.initialize_system(ALLOCATIONS)
    
    # Initial setup
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
    
    # Weekly updates
    for week in range(2):
        results = trader.run_weekly_update()
        
        print(f"\nWeek {week + 1} Results:")
        portfolio = results['portfolio']
        print(f"   Value: ${portfolio['portfolio_value']:,.2f}")
        print(f"   Return: {portfolio['return_pct']:+.2f}%")


if __name__ == "__main__":
    main()