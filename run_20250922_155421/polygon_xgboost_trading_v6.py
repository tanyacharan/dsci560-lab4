#!/usr/bin/env python3
"""
Polygon.io XGBoost HFT Trading System - Final Version
====================================================

This file implements a trading system using Polygon.io's API with 1 minute period data and common indicators (MACD, RSI, BBands) using XGBoost.

New approaches compared to previous versions:
- VWAP only (forward/backward fill)
- No volume features
- All indicators based on VWAP
- Hyper-liquidity assumption (VWAP = execution price)
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
from tqdm import tqdm
import warnings
import os
import logging
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Silence all warnings including XGBoost
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Silence XGBoost internal logging
logging.getLogger('xgboost').setLevel(logging.ERROR)


class TradingConstraints:
    """Trading platform constraints and fees"""

    # Constraints for trading platform Alpaca
    COMMISSION_PER_SHARE = 0.001
    FINRA_TAF_FEE = 0.000166
    SEC_FEE = 0.0000278
    
    MIN_FRACTIONAL_DOLLARS = 1.00
    MIN_FRACTIONAL_SHARES = 0.001
    MAX_SHARES_PER_ORDER = 10000
    
    # HFT parameters
    BASE_STOP_LOSS_BPS = 8  # Increased from 4 to 8
    MAX_STOP_LOSS_BPS = 20  # Increased from 12 to 20
    STOP_LOSS_SCALE = 0.08


class PolygonDataManager:
    """Manages Polygon.io data fetching with VWAP focus"""
    
    def __init__(self, api_key: str):
        # TODO: make .env
        self.client = RESTClient(api_key)
        self.data_cache = {}

    # TODO: Consider sub-1 minute data at some point perhaps?        
    def fetch_ticker_data(self, ticker: str, start_date: datetime, end_date: datetime, 
                         interval: str = "1", timespan: str = "minute") -> pd.DataFrame:
        """Fetch ticker data with VWAP"""
        
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
            
            # Keep ALL trading hours (9:30-16:00) for data completeness
            df = df.set_index('timestamp').between_time("09:30", "16:00").reset_index()
            
            # Ensure we have VWAP - fallback to close if needed
            # TODO: Forgot to check but can VWAP or close exist if there is no trading in that minute?
            if 'vwap' not in df.columns:
                df['vwap'] = df['close']
            
            # Fill missing timestamps and handle gaps
            df = self._fill_missing_timestamps(df)
            
            self.data_cache[cache_key] = df
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _fill_missing_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing timestamps with proper forward/backward fill"""
        if df.empty:
            return df
        
        # Set timestamp as index
        df = df.set_index('timestamp').sort_index()
        
        # Create complete minute-by-minute index for trading hours
        start_date = df.index.min().normalize()
        end_date = df.index.max().normalize() + pd.Timedelta(days=1)
        
        # Generate all trading minutes (9:30-16:00)
        all_minutes = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday-Friday
                day_start = current_date + pd.Timedelta(hours=9, minutes=30)
                day_end = current_date + pd.Timedelta(hours=16)
                day_minutes = pd.date_range(day_start, day_end, freq='1min')[:-1]  # Exclude 16:00 - TODO: Why?
                all_minutes.extend(day_minutes)
            current_date += pd.Timedelta(days=1)
        
        if not all_minutes:
            return df.reset_index()
        
        complete_index = pd.DatetimeIndex(all_minutes)
        complete_index = complete_index[complete_index.isin(df.index.union(complete_index))]
        
        # Reindex with complete trading minutes
        df_complete = df.reindex(complete_index)
        
        # Handle VWAP: forward fill, then backward fill for market open
        # Forward fill recommended *always* unless when market opens at 9:30AM and there's no trade in first minute
        df_complete['vwap'] = df_complete['vwap'].ffill().bfill()
        
        # Handle other price fields the same way
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df_complete.columns:
                df_complete[col] = df_complete[col].ffill().bfill()
        
        # Reset index but preserve timestamp as column
        df_complete = df_complete.reset_index()
        df_complete = df_complete.rename(columns={'index': 'timestamp'})
        return df_complete
    
    def is_trading_time(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within trading hours (10:00-16:00)"""
        time_of_day = timestamp.time()
        return (time_of_day >= pd.Timestamp('10:00').time() and 
                time_of_day < pd.Timestamp('16:00').time() and
                timestamp.weekday() < 5)


class TechnicalIndicators:
    """Calculate technical indicators based on VWAP"""
    
    def __init__(self):
        self.latest_values = {}
        
    def calculate_all_indicators(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate all indicators using VWAP as price"""
        df = df.copy()
        
        if len(df) < 50:  # Need minimum data
            return df
        
        # Use VWAP as our price series
        price_series = df['vwap']
        
        # MACD on VWAP
        df = self._add_macd(df, ticker, price_series)
        
        # RSI on VWAP
        df = self._add_rsi(df, ticker, price_series)
        
        # Bollinger Bands on VWAP
        df = self._add_bollinger_bands(df, ticker, price_series)
        
        # Generate signals
        df = self._generate_signals(df, ticker)
        
        return df
    
    def _add_macd(self, df: pd.DataFrame, ticker: str, price_series: pd.Series, 
                  fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD based on VWAP"""
        ema_fast = price_series.ewm(span=fast, adjust=False).mean()
        ema_slow = price_series.ewm(span=slow, adjust=False).mean()
        
        df[f'{ticker}_macd'] = ema_fast - ema_slow
        df[f'{ticker}_macd_signal'] = df[f'{ticker}_macd'].ewm(span=signal, adjust=False).mean()
        df[f'{ticker}_macd_histogram'] = df[f'{ticker}_macd'] - df[f'{ticker}_macd_signal']
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame, ticker: str, price_series: pd.Series, 
                 window: int = 14) -> pd.DataFrame:
        """RSI based on VWAP"""
        delta = price_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        df[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame, ticker: str, price_series: pd.Series,
                            window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Bollinger Bands based on VWAP"""
        rolling_mean = price_series.rolling(window=window, min_periods=1).mean()
        rolling_std = price_series.rolling(window=window, min_periods=1).std()
        
        df[f'{ticker}_bb_upper'] = rolling_mean + (rolling_std * num_std)
        df[f'{ticker}_bb_middle'] = rolling_mean
        df[f'{ticker}_bb_lower'] = rolling_mean - (rolling_std * num_std)
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Generate trading signals from indicators"""
        # Initialize signals as Hold (0)
        df[f'{ticker}_macd_signal_label'] = 0
        df[f'{ticker}_rsi_signal_label'] = 0
        df[f'{ticker}_bb_signal_label'] = 0
        
        # MACD signals (crossovers)
        macd_valid = df[f'{ticker}_macd'].notna() & df[f'{ticker}_macd_signal'].notna()
        if macd_valid.any():
            macd_cross_up = (df[f'{ticker}_macd'] > df[f'{ticker}_macd_signal']) & \
                           (df[f'{ticker}_macd'].shift(1) <= df[f'{ticker}_macd_signal'].shift(1))
            macd_cross_down = (df[f'{ticker}_macd'] < df[f'{ticker}_macd_signal']) & \
                             (df[f'{ticker}_macd'].shift(1) >= df[f'{ticker}_macd_signal'].shift(1))
            
            df.loc[macd_cross_up & macd_valid, f'{ticker}_macd_signal_label'] = 1   # Buy
            df.loc[macd_cross_down & macd_valid, f'{ticker}_macd_signal_label'] = -1  # Sell
        
        # RSI signals
        rsi_valid = df[f'{ticker}_rsi'].notna()
        if rsi_valid.any():
            df.loc[(df[f'{ticker}_rsi'] < 30) & rsi_valid, f'{ticker}_rsi_signal_label'] = 1   # Oversold
            df.loc[(df[f'{ticker}_rsi'] > 70) & rsi_valid, f'{ticker}_rsi_signal_label'] = -1  # Overbought
        
        # Bollinger Bands signals
        bb_valid = df[f'{ticker}_bb_upper'].notna() & df[f'{ticker}_bb_lower'].notna()
        if bb_valid.any():
            df.loc[(df['vwap'] <= df[f'{ticker}_bb_lower']) & bb_valid, f'{ticker}_bb_signal_label'] = 1   # Buy
            df.loc[(df['vwap'] >= df[f'{ticker}_bb_upper']) & bb_valid, f'{ticker}_bb_signal_label'] = -1  # Sell
        
        return df
    
    def get_current_signals(self, df: pd.DataFrame, ticker: str) -> Dict[str, int]:
        """Get latest signal values"""
        if df.empty:
            return {'macd': 0, 'rsi': 0, 'bollinger': 0}
        
        signals = {}
        for signal_type in ['macd', 'rsi', 'bb']:
            col_name = f'{ticker}_{signal_type}_signal_label'
            if col_name in df.columns:
                value = df[col_name].iloc[-1]
                signals[signal_type.replace('bb', 'bollinger')] = int(value) if not pd.isna(value) else 0
            else:
                signals[signal_type.replace('bb', 'bollinger')] = 0
        
        return signals


class PricePredictionModel:
    """XGBRegressor for VWAP prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.is_trained = False
        self.min_training_samples = 100
        
    def prepare_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare features based on VWAP and indicators"""
        df = df.copy()
        feature_cols = []
        
        # Basic VWAP features
        df['returns'] = df['vwap'].pct_change().fillna(0)
        df['log_returns'] = np.log(df['vwap'] / df['vwap'].shift(1)).fillna(0) # type: ignore - this shouldn't happen?
        feature_cols.extend(['returns', 'log_returns'])
        
        # Technical indicators (fill NaN with neutral values)
        indicator_cols = [col for col in df.columns if ticker in col and any(
            ind in col for ind in ['macd', 'rsi', 'bb_']
        ) and not col.endswith('_signal_label')]
        
        for col in indicator_cols:
            if col in df.columns:
                if 'rsi' in col:
                    df[col] = df[col].fillna(50)  # Neutral RSI
                else:
                    df[col] = df[col].fillna(0)   # Neutral for MACD, BB
                feature_cols.append(col)
        
        # Simple lag features
        for lag in [1, 2]:
            df[f'vwap_lag_{lag}'] = df['vwap'].shift(lag).fillna(method='bfill') # type: ignore - why?
            feature_cols.append(f'vwap_lag_{lag}')
        
        # Price momentum
        df['vwap_momentum_3'] = df['vwap'] - df['vwap'].shift(3)
        df['vwap_momentum_3'] = df['vwap_momentum_3'].fillna(0)
        feature_cols.append('vwap_momentum_3')
        
        self.feature_cols = feature_cols
        return df
    
    def train_model(self, df: pd.DataFrame) -> bool:
        """Train XGBoost on VWAP prediction"""
        # Reset training flag to force retraining
        # This is necessary because of rolling training
        self.is_trained = False
        self.model = None
        
        if len(df) < self.min_training_samples:
            print(f"Insufficient data: {len(df)} rows")
            return False
        
        # Prepare target (next period VWAP)
        df_work = df.copy()
        df_work['target'] = df_work['vwap'].shift(-1)
        
        # Remove last row (no target) and any rows with NaN target
        df_work = df_work[df_work['target'].notna()].copy()
        
        if len(df_work) < self.min_training_samples:
            print(f"Insufficient valid data after target creation: {len(df_work)} rows")
            return False
        
        # Ensure all features exist
        missing_features = [col for col in self.feature_cols if col not in df_work.columns] # type: ignore - prepare_features()
        for col in missing_features:
            df_work[col] = 0
        
        # Final feature cleaning
        for col in self.feature_cols: # type: ignore - prepare_features()
            if df_work[col].isna().any():
                if 'rsi' in col:
                    df_work[col] = df_work[col].fillna(50)
                else:
                    df_work[col] = df_work[col].fillna(0)
        
        X = df_work[self.feature_cols].values
        y = df_work['target'].values
        
        # Train/validation split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Optimized parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4],
            'learning_rate': [0.15, 0.25, 0.4],
            'subsample': [0.75, 0.9],
            'colsample_bytree': [0.75, 0.9]
        }
        
        try:
            # Use CPU for better performance on small datasets
            # TODO: Is there any case where GPU would do better? Experience says no.
            # TODO: Is MSE supposed to be the used here?
            xgb = XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                tree_method='hist',
                device='cpu',
                n_jobs=-10               # Use most cores, leave some for user; user is poor and does not own dedicated processing machine
            )
            print("Using CPU with full parallelization for XGBoost training")
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(
                xgb, param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-5,
                verbose=1  # Show progress since this will take time
            )
            
            grid_search.fit(X_train, y_train)  # type: ignore - somehow bugged?
            self.model = grid_search.best_estimator_
            
            # Validation
            val_pred = self.model.predict(X_val)
            mse = mean_squared_error(y_val, val_pred)  # type: ignore - somehow bugged?
            
            self.is_trained = True
            print(f"Model trained: MSE={mse:.6f}, Features={len(self.feature_cols)}, Data={len(X)}")  # type: ignore - prepare_features()
            print(f"Best params: {grid_search.best_params_}")
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def predict_next_vwap(self, df: pd.DataFrame) -> float:
        """Predict next period VWAP"""
        if not self.is_trained or df.empty:
            return 0.0
        
        try:
            # Get latest row
            latest_row = df.iloc[-1:].copy()
            
            # Ensure all features exist
            for col in self.feature_cols:  # type: ignore - prepare_features()
                if col not in latest_row.columns:
                    latest_row[col] = 0
            
            # Handle NaN
            for col in self.feature_cols:  # type: ignore - prepare_features()
                if latest_row[col].isna().any():
                    if 'rsi' in col:
                        latest_row[col] = latest_row[col].fillna(50)
                    else:
                        latest_row[col] = latest_row[col].fillna(0)
            
            features = latest_row[self.feature_cols].values
            
            if np.isnan(features).any(): # type: ignore - prepare_features()
                return 0.0
            
            # Ensure features are in correct format for prediction
            features = features.reshape(1, -1)  # Ensure 2D array for single prediction | # type: ignore - prepare_features()
            predicted_vwap = self.model.predict(features)[0]  # type: ignore - prepare_features()
            return float(predicted_vwap)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0


class PositionManager:
    """Position management with VWAP-based execution"""
    
    def __init__(self, tickers: List[str], initial_capital: float):
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {ticker: 0 for ticker in tickers}
        self.buy_prices = {ticker: 0.0 for ticker in tickers}  # Average VWAP buy price
        self.max_prices = {ticker: 0.0 for ticker in tickers}
        self.allocations = {}
        self.constraints = TradingConstraints()  # Subject to change for different constraints
        self.pyramid_history = {ticker: 0 for ticker in tickers}
        self.total_fees_paid = 0.0  # Track cumulative transaction fees
        self.ticker_pnl = {ticker: 0.0 for ticker in tickers}  # Track P&L per ticker
        self.weekly_ticker_pnl = {ticker: 0.0 for ticker in tickers}  # Track weekly P&L per ticker
        self.last_trade_pnl = {}  # Store last trade P&L info for display
        self.last_trade_info = {}  # Store last trade info for display
        
    def set_allocations(self, allocations: Optional[Dict[str, float]] = None):
        """Set allocations with 80% capital limit"""
        if allocations:
            total_allocated = sum(allocations.values())
            if total_allocated > 0.8 * self.initial_capital:
                scale_factor = (0.8 * self.initial_capital) / total_allocated
                allocations = {k: v * scale_factor for k, v in allocations.items()}
                print(f"WARNING: Scaled allocations to 80% limit")
            
            self.allocations = allocations
        else:
            allocation_per_ticker = (0.8 * self.initial_capital) / len(self.tickers)
            self.allocations = {ticker: allocation_per_ticker for ticker in self.tickers}
    
    def can_trade_now(self, timestamp: pd.Timestamp) -> bool:
        """Check trading hours (10:00-16:00)"""
        if pd.isna(timestamp):
            return False
        
        time_of_day = timestamp.time()
        is_trading_hours = (time_of_day >= pd.Timestamp('10:00').time() and 
                           time_of_day < pd.Timestamp('16:00').time())
        is_weekday = timestamp.weekday() < 5
        
        return is_trading_hours and is_weekday
    
    def calculate_variable_stop_loss(self, ticker: str, current_vwap: float) -> float:
        """Variable stop loss based on gains"""
        if self.positions[ticker] == 0:
            return 0
        
        bp_gain = ((current_vwap - self.buy_prices[ticker]) / self.buy_prices[ticker]) * 10000
        threshold = self.constraints.BASE_STOP_LOSS_BPS + (bp_gain * self.constraints.STOP_LOSS_SCALE)
        threshold = min(threshold, self.constraints.MAX_STOP_LOSS_BPS)
        
        return max(threshold, self.constraints.BASE_STOP_LOSS_BPS)
    
    def check_safety_net_sell(self, ticker: str, current_vwap: float) -> bool:
        """Safety net auto-sell check"""
        if self.positions[ticker] == 0:
            return False
        
        if current_vwap > self.max_prices[ticker]:
            self.max_prices[ticker] = current_vwap
        
        drop_from_max = ((self.max_prices[ticker] - current_vwap) / self.max_prices[ticker]) * 10000
        threshold = self.calculate_variable_stop_loss(ticker, current_vwap)
        
        if drop_from_max >= threshold:
            return True
        
        return False
    
    def check_end_of_day_sell(self, ticker: str, current_time, current_vwap: float) -> str:
        """Check for end-of-day selling logic"""
        if self.positions[ticker] == 0:
            return "Hold"
        
        time_of_day = current_time.time()
        
        # TODO: Current logic is that sells must happen at 15:50-15:55 due to changes in night market
        # Perhaps we can track whether night market exists and/or prices persist and sell accordingly?
        # Also, do we sell when week ends regardless? Perhaps same logic as above?
        if time_of_day >= pd.Timestamp('15:55').time():  # 3:55 PM - Force sell everything
            return "ForceSell"
        elif time_of_day >= pd.Timestamp('15:50').time():  # 3:50 PM - Cut losses
            if current_vwap <= self.buy_prices[ticker]:  # Position at loss
                return "CutLoss"
        
        return "Hold"
    
    def calculate_position_size(self, ticker: str, current_vwap: float, is_pyramid: bool = False) -> int:
        """Calculate shares to buy"""
        if is_pyramid:
            max_spend = self.allocations[ticker] * 0.75  # 75% for pyramid - TODO: Change this perhaps? Not sure how much should be allocated.
        else:
            max_spend = self.allocations[ticker]
        
        max_spend = min(max_spend, self.current_capital * 0.9)
        shares = int(max_spend / current_vwap)
        
        if shares < self.constraints.MIN_FRACTIONAL_SHARES:
            return 0
        if shares > self.constraints.MAX_SHARES_PER_ORDER:
            shares = self.constraints.MAX_SHARES_PER_ORDER
        
        return shares
    
    # TODO: Extend this for short positions some time soon?
    def calculate_transaction_cost(self, shares: int, is_sell: bool = False) -> float:
        """Transaction costs for both buying and selling"""
        commission = self.constraints.COMMISSION_PER_SHARE * shares
        
        if is_sell:
            finra_taf = self.constraints.FINRA_TAF_FEE * shares
            sec_fee = self.constraints.SEC_FEE * shares
            return commission + finra_taf + sec_fee
        
        return commission
    
    def execute_buy(self, ticker: str, current_vwap: float, is_pyramid: bool = False) -> bool:
        """Execute buy at VWAP (hyper-liquidity assumption)"""
        shares = self.calculate_position_size(ticker, current_vwap, is_pyramid)
        
        if shares == 0:
            return False
        
        transaction_cost = self.calculate_transaction_cost(shares)
        cost = shares * current_vwap + transaction_cost
        
        if cost > self.current_capital:
            return False
        
        self.current_capital -= cost
        self.total_fees_paid += transaction_cost
        
        if self.positions[ticker] == 0:
            # New position
            self.buy_prices[ticker] = current_vwap
            self.max_prices[ticker] = current_vwap
            self.pyramid_history[ticker] = 0
        else:
            # Pyramid: update average buy price
            total_value = (self.positions[ticker] * self.buy_prices[ticker]) + (shares * current_vwap)
            total_shares = self.positions[ticker] + shares
            self.buy_prices[ticker] = total_value / total_shares
            self.pyramid_history[ticker] += 1
        
        self.positions[ticker] += shares
        
        # Store buy info for Trading Activity display
        self.last_trade_info = {
            'shares': shares,
            'is_pyramid': is_pyramid,
            'pyramid_level': self.pyramid_history[ticker] if is_pyramid else 0
        }
        
        return True
    
    def execute_sell(self, ticker: str, current_vwap: float, reason: str = "") -> bool:
        """Execute sell at VWAP"""
        shares = self.positions[ticker]
        
        if shares == 0:
            return False
        
        transaction_cost = self.calculate_transaction_cost(shares, is_sell=True)
        earnings = shares * current_vwap - transaction_cost
        total_cost = shares * self.buy_prices[ticker]
        profit_loss = earnings - total_cost
        pnl_pct = (profit_loss / total_cost) * 100
        
        self.current_capital += earnings
        self.total_fees_paid += transaction_cost
        
        # Track P&L
        self.ticker_pnl[ticker] += profit_loss
        self.weekly_ticker_pnl[ticker] += profit_loss
        
        # Reset position
        self.positions[ticker] = 0
        self.buy_prices[ticker] = 0.0
        self.max_prices[ticker] = 0.0
        self.pyramid_history[ticker] = 0
        
        # Don't print here - will be shown in Trading Activity summary
        # Store P&L info for later display
        self.last_trade_pnl = {
            'profit_loss': profit_loss,
            'pnl_pct': pnl_pct,
            'shares': shares,
            'reason': reason
        }
        
        return True
    
    def reset_weekly_pnl(self):
        """Reset weekly P&L tracking at the start of each week"""
        self.weekly_ticker_pnl = {ticker: 0.0 for ticker in self.tickers}
    
    def get_portfolio_status(self, current_vwaps: Dict[str, float]) -> Dict:
        """Portfolio status using VWAP prices"""
        stock_value = sum(
            self.positions[ticker] * current_vwaps.get(ticker, 0)
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
    """4-signal ensemble for trading decisions"""
    
    def make_decision(self, prediction_signal: int, technical_signals: Dict[str, int], 
                     holding_position: bool, predicted_bp: float = 0) -> str:
        """Make ensemble decision"""
        all_signals = [
            prediction_signal,
            technical_signals['macd'],
            technical_signals['rsi'],
            technical_signals['bollinger']
        ]
        
        buy_count = sum(1 for s in all_signals if s == 1)
        sell_count = sum(1 for s in all_signals if s == -1)
        
        if not holding_position:
            # Need 3/4 buy signals for initial position - TODO: Is this optimal?
            if buy_count >= 3:
                return "Buy"
        else:
            # When holding position
            if sell_count >= 3:
                return "Sell"
            elif prediction_signal == 1 and predicted_bp >= 2.0:
                # Pyramid: +2BP prediction + 2/3 other indicators - TODO: Is this semi-optimal? 
                other_bullish = sum(1 for signal in [
                    technical_signals['macd'],
                    technical_signals['rsi'], 
                    technical_signals['bollinger']
                ] if signal == 1)
                
                if other_bullish >= 2:
                    return "Pyramid"
        
        return "Hold"


class PolygonXGBoostTrader:
    """Main trading system using VWAP"""
    
    def __init__(self, api_key: str, tickers: List[str], initial_capital: float):
        self.data_manager = PolygonDataManager(api_key)
        self.technical_indicators = TechnicalIndicators()
        self.position_manager = PositionManager(tickers, initial_capital)
        self.ensemble_decision = EnsembleDecisionMaker()
        
        self.tickers = tickers
        self.models = {ticker: PricePredictionModel() for ticker in tickers}
        self.performance_history = []
        self.run_folder = None  # Will be set when initializing system
        
    def initialize_system(self, allocations: Optional[Dict[str, float]] = None):
        """Initialize system"""
        # Create main run folder
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_folder = f"run_{run_timestamp}"
        if not os.path.exists(self.run_folder):
            os.makedirs(self.run_folder)
        
        self.position_manager.set_allocations(allocations)
        print(f"VWAP-Based Trading System")
        print(f"   Capital: ${self.position_manager.initial_capital:,.2f}")
        print(f"   Data Collection: 09:30-16:00 EST")
        print(f"   Trading: 10:00-16:00 EST")
        print(f"   Execution: VWAP (hyper-liquidity)")
        print(f"   Tickers: {', '.join(self.tickers)}")
        print(f"   Results will be saved to: {self.run_folder}/")
    
    def fetch_and_prepare_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch data with VWAP focus"""
        prepared_data = {}
        
        for ticker in self.tickers:
            print(f"Processing {ticker}...")
            
            # Fetch with VWAP
            df = self.data_manager.fetch_ticker_data(ticker, start_date, end_date, interval="1")
            
            if df.empty:
                print(f"   No data")
                continue
            
            # Technical indicators on VWAP
            df = self.technical_indicators.calculate_all_indicators(df, ticker)
            
            # ML features
            df = self.models[ticker].prepare_features(df, ticker)
            
            prepared_data[ticker] = df
            print(f"   {len(df)} points | VWAP range: ${df['vwap'].min():.2f}-${df['vwap'].max():.2f}")
        
        return prepared_data
    
    def train_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Train VWAP prediction models"""
        results = {}
        
        for ticker, df in data.items():
            print(f"Training {ticker} VWAP predictor...")
            success = self.models[ticker].train_model(df)
            results[ticker] = success
        
        return results
    
    def execute_trading_cycle(self, current_data: Dict[str, pd.DataFrame]) -> Dict:
        """Execute trades using VWAP"""
        cycle_results = {}
        current_vwaps = {}
        
        # Get current time and VWAP prices
        current_time = None
        for ticker in self.tickers:
            if ticker in current_data and len(current_data[ticker]) > 0:
                current_vwaps[ticker] = current_data[ticker]['vwap'].iloc[-1]
                if current_time is None:
                    current_time = current_data[ticker]['timestamp'].iloc[-1]
        
        # Check trading hours
        if current_time is None or not self.position_manager.can_trade_now(current_time):
            print(f"TIME: Outside trading hours: {current_time}")
            return {'portfolio': self.position_manager.get_portfolio_status(current_vwaps)}
        
        # Execute trades
        for ticker in self.tickers:
            if ticker not in current_data or not self.models[ticker].is_trained:
                continue
            
            df = current_data[ticker]
            current_vwap = current_vwaps[ticker]
            
            # 1. Safety net check if price crashes vs max
            if self.position_manager.check_safety_net_sell(ticker, current_vwap):
                self.position_manager.execute_sell(ticker, current_vwap, "STOP LOSS")
                cycle_results[ticker] = 'SafetyNetSell'
                continue
            
            # 2. Get VWAP prediction
            predicted_vwap = self.models[ticker].predict_next_vwap(df)
            predicted_bp = 0
            prediction_signal = 0
            
            if predicted_vwap > 0:
                predicted_bp = ((predicted_vwap - current_vwap) / current_vwap) * 10000
                if predicted_bp >= 2.0:
                    prediction_signal = 1  # Buy
                elif predicted_bp <= -2.0:
                    prediction_signal = -1  # Sell
            
            # 3. Technical signals
            technical_signals = self.technical_indicators.get_current_signals(df, ticker)
            
            # 4. Ensemble decision
            holding_position = self.position_manager.positions[ticker] > 0
            decision = self.ensemble_decision.make_decision(
                prediction_signal, technical_signals, holding_position, predicted_bp
            )
            
            # Debug output for each ticker
            print(f"  {ticker}: VWAP=${current_vwap:.2f}, Pred={predicted_bp:.1f}BP, Signals={technical_signals}, Decision={decision}")
            
            # 5. Execute
            if decision == "Buy":
                if self.position_manager.execute_buy(ticker, current_vwap):
                    cycle_results[ticker] = 'Buy'
                else:
                    cycle_results[ticker] = 'BuyFailed'
            elif decision == "Sell":
                if self.position_manager.execute_sell(ticker, current_vwap):
                    cycle_results[ticker] = 'Sell'
                else:
                    cycle_results[ticker] = 'SellFailed'
            elif decision == "Pyramid":
                if self.position_manager.execute_buy(ticker, current_vwap, is_pyramid=True):
                    cycle_results[ticker] = 'Pyramid'
                else:
                    cycle_results[ticker] = 'PyramidFailed'
            else:
                cycle_results[ticker] = 'Hold'
        
        # Portfolio status
        portfolio_status = self.position_manager.get_portfolio_status(current_vwaps)
        cycle_results['portfolio'] = portfolio_status
        
        return cycle_results
    
    def run_weekly_update(self) -> Dict:
        """Weekly update: Trade on current week, then retrain for next week"""
        print("\n" + "="*60)
        print("WEEKLY UPDATE")
        print("="*60)
        
        end_date = datetime.now()
        
        # Get 12 weeks of data for training (up to last week)
        training_end = end_date - timedelta(weeks=1)
        training_start = training_end - timedelta(weeks=12)
        
        # Get current week for trading/testing
        trading_start = training_end
        trading_end = end_date
        
        print(f"Training data: {training_start.date()} to {training_end.date()}")
        print(f"Trading data: {trading_start.date()} to {trading_end.date()}")
        
        # Fetch training data (12 weeks, excluding current week)
        training_data = self.fetch_and_prepare_data(training_start, training_end)
        
        # Fetch current week data for trading
        trading_data = self.fetch_and_prepare_data(trading_start, trading_end)
        
        # First: Trade on current week using existing model
        print("\nTrading on current week...")
        trading_results = self.execute_trading_cycle(trading_data)
        
        # Then: Retrain model including current week for next week
        print("\nRetraining models for next week...")
        full_data = self.fetch_and_prepare_data(training_start, end_date)  # Include current week
        training_results = self.train_models(full_data)
        
        # Store performance
        self.performance_history.append({
            'date': datetime.now(),
            'training_results': training_results,
            'trading_results': trading_results,
            'portfolio_value': trading_results['portfolio']['portfolio_value']
        })
        
        # Save weekly results with Monday's date (inside main run folder)
        current_date = datetime.now()
        # Get Monday of current week
        monday = current_date - timedelta(days=current_date.weekday())
        week_folder = f"{self.run_folder}/week_{monday.strftime('%Y%m%d')}"
        
        # Create week directory
        if not os.path.exists(week_folder):
            os.makedirs(week_folder)
        
        # Save training results (model info)
        training_summary = {
            'timestamp': datetime.now().isoformat(),
            'training_results': training_results,
            'model_info': {}
        }
        
        # Get model information for each ticker
        for ticker in self.tickers:
            if self.models[ticker].is_trained:
                model_info = {
                    'feature_count': len(self.models[ticker].feature_cols), # type: ignore - prepare_features()
                    'features': self.models[ticker].feature_cols,
                    'best_params': getattr(self.models[ticker].model, 'get_params', lambda: {})()
                }
                training_summary['model_info'][ticker] = model_info
        
        # Save training summary
        with open(f"{week_folder}/training_summary.json", 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        # Save models (XGBoost uses .json format, not .pth)
        for ticker in self.tickers:
            if self.models[ticker].is_trained:
                self.models[ticker].model.save_model(f"{week_folder}/{ticker}_model.json") # type: ignore - what?
        
        print(f"Week results saved to {week_folder}/")
        return trading_results
    
    def run_backtest(self, start_date: datetime, end_date: datetime, 
                     train_weeks: int = 10, test_weeks: int = 2) -> pd.DataFrame:
        """Run complete backtest with minute-by-minute trading"""
        
        # Calculate dates
        total_weeks = train_weeks + test_weeks
        data_start = start_date - timedelta(weeks=total_weeks)
        
        print(f"Backtest Configuration:")
        print(f"  Data period: {data_start.date()} to {end_date.date()}")
        print(f"  Training: {train_weeks} weeks")
        print(f"  Testing: {test_weeks} weeks")
        print(f"  Test start: {start_date.date()}")
        
        # Fetch all data
        print("\nFetching data...")
        all_data = self.fetch_and_prepare_data(data_start, end_date)
        
        if not all_data:
            print("No data available!")
            return pd.DataFrame()
        
        # Separate training and test data
        training_data = {}
        test_data = {}
        
        for ticker, df in all_data.items():
            if df.empty:
                print(f"  {ticker}: No data available")
                continue
                
            # Check data availability
            data_start = df['timestamp'].min()
            data_end = df['timestamp'].max()
            split_timestamp = pd.Timestamp(start_date)
            
            print(f"  {ticker}: Data from {data_start.date()} to {data_end.date()}")
            
            # If no recent data, use last 20% as test data
            if data_end < split_timestamp:
                print(f"    Warning: No data after {start_date.date()}, using last 20% as test")
                split_idx = int(len(df) * 0.8)
                training_data[ticker] = df.iloc[:split_idx].copy()
                test_data[ticker] = df.iloc[split_idx:].copy()
            else:
                # Normal train/test split
                train_mask = df['timestamp'] < split_timestamp
                test_mask = df['timestamp'] >= split_timestamp
                training_data[ticker] = df[train_mask].copy()
                test_data[ticker] = df[test_mask].copy()
            
            print(f"  {ticker}: {len(training_data[ticker])} train, {len(test_data[ticker])} test points")
        
        # Train models on training data
        print("\nTraining models...")
        training_results = self.train_models(training_data)
        
        # Run backtest on test data
        print("\nRunning minute-by-minute backtest...")
        backtest_results = self._run_minute_by_minute_backtest(test_data)
        
        return backtest_results
    
    def _run_minute_by_minute_backtest(self, test_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Execute trades minute by minute through test period"""
        
        # Get all unique timestamps
        all_timestamps = set()
        for df in test_data.values():
            all_timestamps.update(df['timestamp'].tolist())
        
        # Sort timestamps
        timestamps = sorted(list(all_timestamps))
        
        # Track results
        results = []
        trades_executed = 0
        total_fees_paid = 0.0
        peak_portfolio_value = self.position_manager.initial_capital
        
        # Process each timestamp with tqdm progress bar
        for current_time in tqdm(timestamps, desc="Simulating minutes", unit="min"):
            # Skip if outside trading hours
            if not self.position_manager.can_trade_now(current_time):
                continue
            
            # Get data up to current time for each ticker
            current_data = {}
            current_vwaps = {}
            
            for ticker, df in test_data.items():
                # Get all data up to and including current time
                mask = df['timestamp'] <= current_time
                current_df = df[mask].copy()
                
                if len(current_df) == 0:
                    continue
                
                current_data[ticker] = current_df
                current_vwaps[ticker] = current_df['vwap'].iloc[-1]
            
            # Make trading decisions for each ticker
            minute_results = {
                'timestamp': current_time,
                'portfolio_value': 0,
                'cash': self.position_manager.current_capital
            }
            
            for ticker in self.tickers:
                if ticker not in current_data or not self.models[ticker].is_trained:
                    continue
                
                df = current_data[ticker]
                current_vwap = current_vwaps[ticker]
                
                # Store current position
                minute_results[f'{ticker}_position'] = self.position_manager.positions[ticker]
                minute_results[f'{ticker}_vwap'] = current_vwap
                
                # Get prediction and signals FIRST (for all actions including SafetyNetSell)
                predicted_vwap = self.models[ticker].predict_next_vwap(df)
                predicted_bp = 0
                prediction_signal = 0
                
                if predicted_vwap > 0:
                    predicted_bp = ((predicted_vwap - current_vwap) / current_vwap) * 10000
                    minute_results[f'{ticker}_pred_bp'] = predicted_bp
                    
                    if predicted_bp >= 2.0:
                        prediction_signal = 1
                    elif predicted_bp <= -2.0:
                        prediction_signal = -1
                
                # Get technical signals
                technical_signals = self.technical_indicators.get_current_signals(df, ticker)
                minute_results[f'{ticker}_signals'] = f"P:{prediction_signal},M:{technical_signals['macd']},R:{technical_signals['rsi']},B:{technical_signals['bollinger']}"
                
                # 1. Safety net check
                if self.position_manager.check_safety_net_sell(ticker, current_vwap):
                    if self.position_manager.execute_sell(ticker, current_vwap, "STOP LOSS"):
                        minute_results[f'{ticker}_action'] = 'SafetyNetSell'
                        trades_executed += 1
                        continue
                
                # 2. End-of-day selling logic
                eod_decision = self.position_manager.check_end_of_day_sell(ticker, current_time, current_vwap)
                if eod_decision in ['ForceSell', 'CutLoss']:
                    reason = "FORCE SELL" if eod_decision == 'ForceSell' else "CUT LOSS"
                    if self.position_manager.execute_sell(ticker, current_vwap, reason):
                        minute_results[f'{ticker}_action'] = eod_decision
                        trades_executed += 1
                        continue
                
                # 3. Check if we're in end-of-day lockout period (no new purchases after 3:50 PM)
                time_of_day = current_time.time()
                if time_of_day >= pd.Timestamp('15:50').time():
                    minute_results[f'{ticker}_action'] = 'Hold'
                    continue
                
                # 4. Ensemble decision (using already calculated prediction/signals)
                holding_position = self.position_manager.positions[ticker] > 0
                decision = self.ensemble_decision.make_decision(
                    prediction_signal, technical_signals, holding_position, predicted_bp
                )
                
                minute_results[f'{ticker}_decision'] = decision
                
                # 5. Execute trade
                if decision == "Buy":
                    if self.position_manager.execute_buy(ticker, current_vwap):
                        minute_results[f'{ticker}_action'] = 'Buy'
                        trades_executed += 1
                elif decision == "Sell":
                    if self.position_manager.execute_sell(ticker, current_vwap):
                        minute_results[f'{ticker}_action'] = 'Sell'
                        trades_executed += 1
                elif decision == "Pyramid":
                    if self.position_manager.execute_buy(ticker, current_vwap, is_pyramid=True):
                        minute_results[f'{ticker}_action'] = 'Pyramid'
                        trades_executed += 1
                else:
                    minute_results[f'{ticker}_action'] = 'Hold'
            
            # Calculate portfolio value
            portfolio = self.position_manager.get_portfolio_status(current_vwaps)
            minute_results['portfolio_value'] = portfolio['portfolio_value']
            minute_results['return_pct'] = portfolio['return_pct']
            
            # Comprehensive portfolio tracking
            current_cash = self.position_manager.current_capital
            total_portfolio_value = portfolio['portfolio_value']
            total_fees = self.position_manager.total_fees_paid
            profit = total_portfolio_value - self.position_manager.initial_capital
            pure_profit = profit + total_fees  # What profit would be with 0 fees
            
            # Update peak for drawdown calculation
            # TODO: I only added this because it might be helpful - what is this?
            peak_portfolio_value = max(peak_portfolio_value, total_portfolio_value)
            drawdown = (peak_portfolio_value - total_portfolio_value) / peak_portfolio_value * 100
            
            # Position metrics
            total_shares = sum(self.position_manager.positions.values())
            active_positions = sum(1 for pos in self.position_manager.positions.values() if pos > 0)
            capital_utilization = (total_portfolio_value - current_cash) / total_portfolio_value * 100 if total_portfolio_value > 0 else 0
            
            # Add comprehensive tracking
            minute_results.update({
                'timestamp_str': current_time.strftime('%Y-%m-%d %H:%M'),
                'current_cash': current_cash,
                'total_fees_paid': total_fees,
                'profit': profit,
                'pure_profit': pure_profit,
                'total_shares_owned': total_shares,
                'active_positions': active_positions,
                'capital_utilization': capital_utilization,
                'drawdown_pct': drawdown,
                'total_trades': trades_executed
            })
            
            # Real-time trading activity - prints above tqdm progress bar
            minute_actions = []
            for ticker in self.tickers:
                action = minute_results.get(f'{ticker}_action', 'Hold')
                if action != 'Hold':
                    pred_bp = minute_results.get(f'{ticker}_pred_bp', 0)
                    signals = minute_results.get(f'{ticker}_signals', 'N/A')
                    
                    # Add P&L info for sell actions or shares info for buy actions
                    extra_str = ""
                    if action in ['Sell', 'SafetyNetSell', 'ForceSell', 'CutLoss'] and hasattr(self.position_manager, 'last_trade_pnl'):
                        pnl_info = self.position_manager.last_trade_pnl
                        profit_loss = pnl_info['profit_loss']
                        pnl_pct = pnl_info['pnl_pct']
                        reason = pnl_info['reason']
                        result = "PROFIT" if profit_loss > 0 else "LOSS"
                        reason_str = f" ({reason})" if reason else ""
                        extra_str = f" | {result}{reason_str}: ${profit_loss:+,.2f} ({pnl_pct:+.2f}%)"
                    elif action in ['Buy', 'Pyramid'] and hasattr(self.position_manager, 'last_trade_info'):
                        trade_info = self.position_manager.last_trade_info
                        shares = trade_info.get('shares', 0)
                        if action == 'Pyramid':
                            pyramid_level = trade_info.get('pyramid_level', 0)
                            extra_str = f" | {shares} shares (Pyramid #{pyramid_level})"
                        else:
                            extra_str = f" | {shares} shares"
                    
                    minute_actions.append(f"{ticker}: {action} @${current_vwaps.get(ticker, 0):.2f} (Pred={pred_bp:.1f}BP, Signals={signals}){extra_str}")
            
            if minute_actions:
                # tqdm.write() prints above the progress bar without interfering
                tqdm.write(f"{current_time.strftime('%Y-%m-%d %H:%M')} - Trading Activity:")
                for action in minute_actions:
                    tqdm.write(f"   {action}")
                portfolio = self.position_manager.get_portfolio_status(current_vwaps)
                tqdm.write(f"   Portfolio: ${portfolio['portfolio_value']:,.0f} ({portfolio['return_pct']:+.2f}%)")
            
            results.append(minute_results)
        
        print(f"\nBacktest complete:")
        print(f"  Total trades executed: {trades_executed}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Summary statistics
        if len(results_df) > 0:
            final_value = results_df['portfolio_value'].iloc[-1]
            final_return = results_df['return_pct'].iloc[-1]
            max_value = results_df['portfolio_value'].max()
            min_value = results_df['portfolio_value'].min()
            
            # Additional metrics
            final_cash = results_df['current_cash'].iloc[-1]
            total_fees = results_df['total_fees_paid'].iloc[-1]
            profit = results_df['profit'].iloc[-1]
            pure_profit = results_df['pure_profit'].iloc[-1]
            max_drawdown = results_df['drawdown_pct'].max()
            avg_positions = results_df['active_positions'].mean()
            avg_capital_used = results_df['capital_utilization'].mean()
            
            print(f"  Final portfolio value: ${final_value:,.2f}")
            print(f"  Final return: {final_return:.2f}%")
            print(f"  Current cash: ${final_cash:,.2f}")
            print(f"  Total fees paid: ${total_fees:,.2f}")
            print(f"  Net profit: ${profit:,.2f}")
            print(f"  Pure profit (no fees): ${pure_profit:,.2f}")
            print(f"  Fee impact: ${total_fees:,.2f} ({(total_fees/profit*100) if profit > 0 else 0:.1f}% of profit)")
            print(f"  Max portfolio value: ${max_value:,.2f}")
            print(f"  Min portfolio value: ${min_value:,.2f}")
            print(f"  Max drawdown: {max_drawdown:.2f}%")
            print(f"  Avg active positions: {avg_positions:.1f}")
            print(f"  Avg capital utilization: {avg_capital_used:.1f}%")
            
            # Show sample of trading activity
            action_cols = [col for col in results_df.columns if '_action' in col]
            trades = results_df[results_df[action_cols].apply(lambda x: x.notna() & (x != 'Hold')).any(axis=1)]
            
            if not trades.empty:
                print(f"\nFirst 10 trades:")
                
                # Reshape data to ticker, pred_bp, action format (exclude Holds)
                clean_trades = []
                for _, row in trades.head(10).iterrows():
                    timestamp = row['timestamp_str']
                    for ticker in self.tickers:
                        action = row.get(f'{ticker}_action')
                        pred_bp = row.get(f'{ticker}_pred_bp')
                        
                        if pd.notna(action) and action != 'Hold':  # Exclude Hold actions
                            clean_trades.append({
                                'timestamp': timestamp,
                                'ticker': ticker,
                                'pred_bp': pred_bp,
                                'action': action
                            })
                
                if clean_trades:
                    clean_df = pd.DataFrame(clean_trades)
                    print(clean_df.to_string(index=False))
        
        return results_df

# TODO: Connect this to portfolio system later?
def main():
    """Clean VWAP-based trading system"""
    # Configuration  
    API_KEY = os.getenv("POLYGON_API_KEY")
    INITIAL_CAPITAL = 500000
    # Full ticker list for production
    # TODO: "MCD", "PEP" are removed due to insanely chaotic predictions for XGBoost - perhaps intraday behavior?
    # Consider adapting this for future reference
    TICKERS = ["TSLA", "NVDA", "AAPL", "META", "PFE", "PLTR", "UBER", "BAC", "V", "NFLX"]
    
    # Test with just 2 tickers for faster training
    # TICKERS = ["AAPL", "META"]
    
    # Allocations for test tickers
    # ALLOCATIONS = {
    #     "AAPL": 25000,      # Tech
    #     "META": 25000,      # Tech
    # }
    
    # Dynamic allocation will be calculated in weekly update using current cash
    # Tech: 6%, Non-tech: 9%
    # Note: 2.5x buff for now due to low cap% - TODO: Change this later?
    TECH_ALLOCATION_PCT = 0.15
    NON_TECH_ALLOCATION_PCT = 0.225
    
    # Initialize
    trader = PolygonXGBoostTrader(
        api_key=API_KEY,
        tickers=TICKERS,
        initial_capital=INITIAL_CAPITAL
    )
    
    # Calculate initial dynamic allocations
    current_cash = trader.position_manager.current_capital
    tech_tickers = ["TSLA", "NVDA", "AAPL", "META", "PLTR", "NFLX"]
    
    allocations = {}
    for ticker in TICKERS:
        if ticker in tech_tickers:
            allocations[ticker] = TECH_ALLOCATION_PCT * current_cash
        else:
            allocations[ticker] = NON_TECH_ALLOCATION_PCT * current_cash
    
    trader.initialize_system(allocations)
    
    # Run rolling weekly backtest from Jan 5 to Sep 19
    print("\nRUNNING ROLLING WEEKLY BACKTEST")
    start_date = datetime(2025, 1, 5)   # First test week starts Jan 5, 2025
    end_date = datetime(2025, 9, 19)    # TODO: Testing - change to Sep 19, 2025
    
    # Create cumulative stats file
    stats_file = f"{trader.run_folder}/weekly_cumulative_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("CUMULATIVE WEEKLY PERFORMANCE STATS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Start Date: {start_date.date()}\n")
        f.write(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}\n")
        f.write("=" * 60 + "\n\n")
    
    # Initialize historical data CSV files (one per ticker)
    data_files = {}
    for ticker in TICKERS:
        data_files[ticker] = f"{trader.run_folder}/historical_data_{ticker}.csv"
    
    # Initial data fetch: Get 12 weeks before first test week
    initial_train_end = start_date - timedelta(days=1)  # Jan 4, 2025
    initial_train_start = initial_train_end - timedelta(weeks=12)  # ~Oct 2024
    
    print(f"\nInitial data fetch: {initial_train_start.date()} to {initial_train_end.date()}")
    initial_data = trader.fetch_and_prepare_data(initial_train_start, initial_train_end)
    
    # Save initial data to CSV files
    for ticker, df in initial_data.items():
        if not df.empty:
            df.to_csv(data_files[ticker], index=False)
            print(f"Saved {len(df)} rows for {ticker} to {data_files[ticker]}")
    
    # Run rolling weekly simulation
    current_week_start = start_date
    week_num = 1
    previous_week_ending_value = INITIAL_CAPITAL  # Track previous week's ending value for week-over-week profit
    
    # Track per-ticker P&L
    ticker_pnl_tracking = {ticker: [] for ticker in TICKERS}
    
    while current_week_start < end_date:
        current_week_end = current_week_start + timedelta(days=6)
        if current_week_end > end_date:
            current_week_end = end_date
        
        print(f"\n{'='*60}")
        print(f"WEEK {week_num}: {current_week_start.date()} to {current_week_end.date()}")
        print(f"{'='*60}")
        
        # Load historical data from CSV files and select most recent 12 weeks for training
        print(f"Loading historical data from CSV files...")
        training_data = {}
        train_cutoff = current_week_start - timedelta(weeks=12)  # 12 weeks before current week
        
        for ticker in TICKERS:
            if os.path.exists(data_files[ticker]):
                df = pd.read_csv(data_files[ticker])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Select only the most recent 12 weeks for training
                training_df = df[df['timestamp'] >= train_cutoff].copy()
                training_df = training_df[training_df['timestamp'] < current_week_start]  # Exclude current week
                
                training_data[ticker] = training_df
                print(f"  Loaded {len(df)} total rows for {ticker}, using {len(training_df)} rows for training (last 12 weeks)")
            else:
                training_data[ticker] = pd.DataFrame()
        
        # Train models with existing historical data
        training_results = trader.train_models(training_data)
        
        # Fetch ONLY current week's data via API call (no cheating!)
        print(f"Fresh API call: Fetching current week data from {current_week_start.date()} to {current_week_end.date()}")
        trader.data_manager.data_cache.clear()  # Force fresh API call
        test_data = trader.fetch_and_prepare_data(current_week_start, current_week_end)
        
        # Run minute-by-minute backtest for this week
        week_results = trader._run_minute_by_minute_backtest(test_data)
        
        # Update capital allocations based on current cash
        current_cash = trader.position_manager.current_capital
        allocations = {}
        for ticker in TICKERS:
            if ticker in tech_tickers:
                allocations[ticker] = TECH_ALLOCATION_PCT * current_cash
            else:
                allocations[ticker] = NON_TECH_ALLOCATION_PCT * current_cash
        trader.position_manager.allocations = allocations
        
        # Save week results
        if not week_results.empty:
            week_folder = f"{trader.run_folder}/week_{current_week_start.strftime('%Y%m%d')}"
            os.makedirs(week_folder, exist_ok=True)
            week_results.to_csv(f"{week_folder}/trading_results.csv", index=False)
            
            # Calculate week stats
            initial_week_value = week_results['portfolio_value'].iloc[0] if len(week_results) > 0 else trader.position_manager.initial_capital
            final_week_value = week_results['portfolio_value'].iloc[-1] if len(week_results) > 0 else initial_week_value
            week_return = (final_week_value - initial_week_value) / initial_week_value * 100
            week_trades = week_results['total_trades'].iloc[-1] if len(week_results) > 0 else 0
            week_fees = week_results['total_fees_paid'].iloc[-1] - (week_results['total_fees_paid'].iloc[0] if len(week_results) > 0 else 0)
            
            # Calculate weekly profit (from previous week's ending value)
            weekly_profit = final_week_value - previous_week_ending_value
            weekly_profit_pct = (weekly_profit / previous_week_ending_value) * 100
            
            # Get per-ticker P&L from PositionManager
            ticker_weekly_pnl = trader.position_manager.weekly_ticker_pnl.copy()
            
            # Print weekly summary to console
            print(f"\n{'='*60}")
            print(f"WEEK {week_num} SUMMARY")
            print(f"{'='*60}")
            print(f"Starting Portfolio Value: ${initial_week_value:,.2f}")
            print(f"Ending Portfolio Value: ${final_week_value:,.2f}")
            print(f"Week Return: {week_return:+.2f}%")
            print(f"Weekly Profit: ${weekly_profit:,.2f} ({weekly_profit_pct:+.2f}%)")
            print(f"Total Trades: {week_trades}")
            print(f"Week Fees: ${week_fees:,.2f}")
            print(f"\nPer-Ticker P&L:")
            for ticker in sorted(TICKERS):
                ticker_pnl = ticker_weekly_pnl.get(ticker, 0)
                if ticker_pnl != 0:  # Only show tickers with activity
                    print(f"  {ticker}: ${ticker_pnl:+,.2f}")
            print(f"{'='*60}")
            
            # Append to cumulative stats file
            with open(stats_file, 'a') as f:
                f.write(f"\nWEEK {week_num}: {current_week_start.date()} to {current_week_end.date()}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Starting Portfolio Value: ${initial_week_value:,.2f}\n")
                f.write(f"Ending Portfolio Value: ${final_week_value:,.2f}\n")
                f.write(f"Week Return: {week_return:+.2f}%\n")
                f.write(f"Weekly Profit: ${weekly_profit:,.2f} ({weekly_profit_pct:+.2f}%)\n")
                f.write(f"Total Trades: {week_trades}\n")
                f.write(f"Week Fees: ${week_fees:,.2f}\n")
                f.write(f"Current Cash: ${trader.position_manager.current_capital:,.2f}\n")
                f.write(f"Active Positions: {sum(1 for pos in trader.position_manager.positions.values() if pos > 0)}\n")
                f.write(f"Cumulative Return: {((final_week_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100):+.2f}%\n")
                
                # Write per-ticker P&L
                f.write(f"\nPer-Ticker P&L:\n")
                for ticker in sorted(TICKERS):
                    ticker_pnl = ticker_weekly_pnl.get(ticker, 0)
                    f.write(f"  {ticker}: ${ticker_pnl:,.2f}\n")
                f.write("-" * 40 + "\n")
        
        # After trading, append new week to historical data files (keep ALL data)
        print(f"\nAppending new week to historical data files...")
        
        for ticker in TICKERS:
            if ticker in test_data and not test_data[ticker].empty:
                # Load existing data
                if os.path.exists(data_files[ticker]):
                    historical_df = pd.read_csv(data_files[ticker])
                    historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
                else:
                    historical_df = pd.DataFrame()
                
                # Append new week's data
                combined_df = pd.concat([historical_df, test_data[ticker]], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                combined_df = combined_df.sort_values('timestamp')
                
                # Save ALL data (no deletion)
                combined_df.to_csv(data_files[ticker], index=False)
                print(f"  Updated {ticker}: Added {len(test_data[ticker])} rows, total {len(combined_df)} rows")
        
        # Update previous week's ending value for next week's profit calculation
        if not week_results.empty:
            previous_week_ending_value = week_results['portfolio_value'].iloc[-1]
        
        # Reset weekly P&L tracker for next week
        trader.position_manager.reset_weekly_pnl()
        
        # Move to next week
        current_week_start = current_week_end + timedelta(days=1)
        week_num += 1
    
    print(f"\nWeekly stats saved to: {stats_file}")
    
    # Save final configuration
    final_config = {
        'timestamp': datetime.now().isoformat(),
        'tickers': trader.tickers,
        'initial_capital': trader.position_manager.initial_capital,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_weeks': week_num - 1,
        'tech_allocation_pct': TECH_ALLOCATION_PCT,
        'non_tech_allocation_pct': NON_TECH_ALLOCATION_PCT,
        'final_portfolio_value': float(trader.position_manager.get_portfolio_status({})['portfolio_value']),
        'final_cash': float(trader.position_manager.current_capital)
    }
    
    with open(f'{trader.run_folder}/rolling_backtest_config.json', 'w') as f:
        json.dump(final_config, f, indent=2)
    
    print(f"\nAll results saved to {trader.run_folder}/")
    print(f"  - weekly_cumulative_stats.txt (cumulative weekly performance)")
    print(f"  - week_YYYYMMDD/ folders (individual week data)")
    print(f"  - rolling_backtest_config.json (configuration and final summary)")
    print(f"  - historical_data_TICKER.csv files (accumulated price data)")


if __name__ == "__main__":
    main()