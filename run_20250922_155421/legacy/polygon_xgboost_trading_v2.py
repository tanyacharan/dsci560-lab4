#!/usr/bin/env python3
"""
Polygon.io XGBoost HFT Trading System with Capital Management
============================================================

Complete trading system with:
- Polygon.io data integration
- Technical indicators (MACD, RSI, Bollinger Bands)
- XGBoost with GridSearchCV
- Capital management and position sizing
- Pyramid trading for trending markets
- Stop loss for risk management (4 basis points)
- Transaction cost calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, List, Tuple
from polygon import RESTClient
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import warnings
warnings.filterwarnings('ignore')


class AlpacaConstraints:
    """Trading platform constraints and fees"""
    COMMISSION_PER_SHARE = 0.001  # Under 10M shares/month
    FINRA_TAF_FEE = 0.000166  # Per share
    SEC_FEE = 0.0000278  # Per share
    
    # Position constraints  
    MIN_FRACTIONAL_DOLLARS = 1.00
    MIN_FRACTIONAL_SHARES = 0.001
    MAX_SHARES_PER_ORDER = 10000
    
    # HFT parameters
    STOP_LOSS_BPS = 4  # 4 basis points for HFT
    

class PolygonDataManager:
    """Manages Polygon.io data fetching and processing"""
    
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)
        self.data_cache = {}
        
    def fetch_ticker_data(self, ticker: str, start_date: datetime, end_date: datetime, 
                         interval: str = "5", timespan: str = "minute") -> pd.DataFrame:
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
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Keep only regular trading hours
            df = df.set_index('timestamp').between_time("09:30", "16:00").reset_index()
            
            self.data_cache[cache_key] = df
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()


class TechnicalAnalysis:
    """Technical indicators calculated as instance methods for better integration"""
    
    def __init__(self):
        self.indicator_values = {}
        
    def calculate_indicators(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate all technical indicators for a ticker"""
        df = df.copy()
        
        # MACD
        macd_data = self.calculate_macd(df['close'])
        df[f'{ticker}_macd'] = macd_data['macd']
        df[f'{ticker}_macd_signal'] = macd_data['signal']
        df[f'{ticker}_macd_histogram'] = macd_data['histogram']
        
        # RSI
        df[f'{ticker}_rsi'] = self.calculate_rsi(df['close'])
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df['close'])
        df[f'{ticker}_bb_upper'] = bb_data['upper']
        df[f'{ticker}_bb_middle'] = bb_data['middle']
        df[f'{ticker}_bb_lower'] = bb_data['lower']
        
        # Store latest values for quick access
        self.indicator_values[ticker] = {
            'macd': df[f'{ticker}_macd'].iloc[-1],
            'macd_signal': df[f'{ticker}_macd_signal'].iloc[-1],
            'rsi': df[f'{ticker}_rsi'].iloc[-1],
            'bb_upper': df[f'{ticker}_bb_upper'].iloc[-1],
            'bb_lower': df[f'{ticker}_bb_lower'].iloc[-1],
            'close': df['close'].iloc[-1]
        }
        
        # Generate signals
        df[f'{ticker}_signal'] = self.generate_signals(df, ticker)
        
        return df
    
    def calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, series: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return {
            'upper': upper_band,
            'middle': rolling_mean,
            'lower': lower_band
        }
    
    def generate_signals(self, df: pd.DataFrame, ticker: str) -> pd.Series:
        """Generate trading signals using voting system"""
        # MACD signal
        macd_signal = pd.Series(0, index=df.index)  # 0 = Hold
        macd_buy = (df[f'{ticker}_macd'] > df[f'{ticker}_macd_signal']) & \
                   (df[f'{ticker}_macd'].shift(1) <= df[f'{ticker}_macd_signal'].shift(1))
        macd_sell = (df[f'{ticker}_macd'] < df[f'{ticker}_macd_signal']) & \
                    (df[f'{ticker}_macd'].shift(1) >= df[f'{ticker}_macd_signal'].shift(1))
        macd_signal[macd_buy] = 1   # Buy
        macd_signal[macd_sell] = -1  # Sell
        
        # RSI signal
        rsi_signal = pd.Series(0, index=df.index)
        rsi_signal[df[f'{ticker}_rsi'] < 30] = 1   # Oversold = Buy
        rsi_signal[df[f'{ticker}_rsi'] > 70] = -1  # Overbought = Sell
        
        # Bollinger Bands signal
        bb_signal = pd.Series(0, index=df.index)
        bb_signal[df['close'] <= df[f'{ticker}_bb_lower']] = 1   # Buy
        bb_signal[df['close'] >= df[f'{ticker}_bb_upper']] = -1  # Sell
        
        # Voting system: majority wins
        total_signal = macd_signal + rsi_signal + bb_signal
        final_signal = pd.Series(0, index=df.index)
        final_signal[total_signal >= 2] = 1    # Buy
        final_signal[total_signal <= -2] = -1  # Sell
        
        return final_signal


class PositionManager:
    """Manages capital allocation, positions, and trade execution"""
    
    def __init__(self, tickers: List[str], initial_capital: float):
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {ticker: 0 for ticker in tickers}  # Current shares held
        self.buy_prices = {ticker: 0.0 for ticker in tickers}  # Average buy price
        self.allocations = {}  # Max capital per ticker
        self.position_levels = {ticker: [] for ticker in tickers}  # For pyramid trading
        self.constraints = AlpacaConstraints()
        
    def set_allocations(self, allocations: Optional[Dict[str, float]] = None):
        """Set capital allocations for each ticker"""
        if allocations:
            self.allocations = allocations
        else:
            # Equal allocation
            allocation_per_ticker = self.initial_capital / len(self.tickers)
            self.allocations = {ticker: allocation_per_ticker for ticker in self.tickers}
    
    def check_stop_loss(self, ticker: str, current_price: float) -> bool:
        """Check if stop loss is triggered (4 basis points for HFT)"""
        if self.positions[ticker] == 0 or self.buy_prices[ticker] == 0:
            return False
        
        buy_price = self.buy_prices[ticker]
        price_drop_bps = ((buy_price - current_price) / buy_price) * 10000
        
        if price_drop_bps >= self.constraints.STOP_LOSS_BPS:
            print(f"Stop loss triggered for {ticker}: {price_drop_bps:.1f} bps drop")
            return True
        return False
    
    def should_pyramid_buy(self, ticker: str, current_price: float, predicted_signal: int) -> bool:
        """Check if we should add to existing position (pyramid trading)"""
        if self.positions[ticker] == 0 or predicted_signal != 1:
            return False
        
        # Check if price has risen enough since last buy (5 basis points)
        last_buy_price = self.position_levels[ticker][-1][0] if self.position_levels[ticker] else self.buy_prices[ticker]
        bp_increase = ((current_price - last_buy_price) / last_buy_price) * 10000
        
        return bp_increase >= 5  # Pyramid if 5+ basis points increase
    
    def calculate_position_size(self, ticker: str, current_price: float, is_pyramid: bool = False) -> int:
        """Calculate how many shares to buy"""
        if is_pyramid:
            # Pyramid buy: use 25% of remaining capital or 50% of allocation
            max_spend = min(self.current_capital * 0.25, self.allocations[ticker] * 0.5)
        else:
            # Initial position: use full allocation
            max_spend = min(self.allocations[ticker], self.current_capital)
        
        shares = int(max_spend / current_price)
        
        # Check constraints
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
        else:
            return commission
    
    def execute_buy(self, ticker: str, current_price: float, is_pyramid: bool = False) -> bool:
        """Execute buy order"""
        shares = self.calculate_position_size(ticker, current_price, is_pyramid)
        
        if shares == 0:
            print(f"Cannot buy {ticker}: insufficient funds or constraints violated")
            return False
        
        cost = shares * current_price + self.calculate_transaction_cost(shares)
        
        if cost > self.current_capital:
            print(f"Cannot buy {ticker}: total cost ${cost:.2f} > available ${self.current_capital:.2f}")
            return False
        
        # Execute trade
        self.current_capital -= cost
        
        # Update position tracking
        if self.positions[ticker] == 0:
            self.buy_prices[ticker] = current_price
        else:
            # Update average buy price
            total_value = (self.positions[ticker] * self.buy_prices[ticker]) + (shares * current_price)
            self.positions[ticker] += shares
            self.buy_prices[ticker] = total_value / self.positions[ticker]
        
        self.positions[ticker] += shares
        self.position_levels[ticker].append((current_price, shares))
        
        action = "Pyramid buy" if is_pyramid else "Initial buy"
        print(f"{action} {ticker}: {shares} shares @ ${current_price:.2f} (Total: {self.positions[ticker]} shares)")
        
        if self.current_capital <= 0.1 * self.initial_capital:
            print("Warning: Low capital remaining")
        
        return True
    
    def execute_sell(self, ticker: str, current_price: float) -> bool:
        """Execute sell order (sell all shares)"""
        shares = self.positions[ticker]
        
        if shares == 0:
            print(f"No position to sell for {ticker}")
            return False
        
        # Calculate earnings
        earnings = shares * current_price - self.calculate_transaction_cost(shares, is_sell=True)
        
        # Calculate P&L
        total_cost = shares * self.buy_prices[ticker]
        profit_loss = earnings - total_cost
        pnl_pct = (profit_loss / total_cost) * 100
        
        # Execute trade
        self.current_capital += earnings
        self.positions[ticker] = 0
        self.buy_prices[ticker] = 0.0
        self.position_levels[ticker] = []
        
        emoji = "PROFIT" if profit_loss > 0 else "LOSS"
        print(f"{emoji} Sold all {shares} shares of {ticker} @ ${current_price:.2f}")
        print(f"   P&L: ${profit_loss:.2f} ({pnl_pct:+.2f}%)")
        
        return True
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        stock_value = sum(
            self.positions[ticker] * current_prices.get(ticker, 0)
            for ticker in self.tickers
        )
        return self.current_capital + stock_value
    
    def get_portfolio_status(self, current_prices: Dict[str, float]) -> Dict:
        """Get detailed portfolio status"""
        portfolio_value = self.get_portfolio_value(current_prices)
        
        return {
            'portfolio_value': portfolio_value,
            'current_capital': self.current_capital,
            'positions': self.positions.copy(),
            'allocations': self.allocations.copy(),
            'return_pct': ((portfolio_value - self.initial_capital) / self.initial_capital) * 100,
            'capital_usage': 1 - (self.current_capital / self.initial_capital)
        }


class XGBoostHFTModel:
    """XGBoost model optimized for HFT with proper label mapping"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_cols = None
        self.label_map = {'Sell': -1, 'Hold': 0, 'Buy': 1}  # Changed to -1, 0, 1
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.bp_threshold_buy = 10   # Adaptive threshold for buy signals
        self.bp_threshold_sell = 4   # Adaptive threshold for sell signals
        
    def prepare_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare features for XGBoost"""
        df = df.copy()
        feature_cols = []
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_momentum'] = df['close'] - df['close'].shift(5)
        df['bp_change'] = (df['close'].pct_change() * 10000)  # Basis points change
        feature_cols.extend(['returns', 'log_returns', 'price_momentum', 'bp_change'])
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['dollar_volume'] = df['close'] * df['volume']
        feature_cols.extend(['volume_ratio', 'dollar_volume'])
        
        # Technical indicators (already calculated)
        tech_cols = [col for col in df.columns if ticker in col and any(
            ind in col for ind in ['macd', 'rsi', 'bb_']
        )]
        feature_cols.extend(tech_cols)
        
        # Microstructure features for HFT
        df['bid_ask_spread'] = df['high'] - df['low']  # Proxy for spread
        df['price_efficiency'] = df['close'] / df['vwap'] if 'vwap' in df.columns else 1
        feature_cols.extend(['bid_ask_spread', 'price_efficiency'])
        
        # Lag features (important for time series)
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'bp_change_lag_{lag}'] = df['bp_change'].shift(lag)
            feature_cols.extend([f'close_lag_{lag}', f'volume_lag_{lag}', f'bp_change_lag_{lag}'])
        
        self.feature_cols = feature_cols
        return df
    
    def adapt_thresholds(self, performance: str):
        """Adapt basis point thresholds based on performance"""
        if performance == "profitable":
            # Raise the bar
            self.bp_threshold_buy = min(self.bp_threshold_buy + 1, 30)
            self.bp_threshold_sell = min(self.bp_threshold_sell + 0.5, 15)
        elif performance == "loss":
            # Be less selective
            self.bp_threshold_buy = max(self.bp_threshold_buy - 1, 5)
            self.bp_threshold_sell = max(self.bp_threshold_sell - 0.5, 2)
    
    def grid_search_train(self, X_train: np.ndarray, y_train: np.ndarray, 
                         cv_splits: int = 5) -> Dict:
        """Train with GridSearchCV using TimeSeriesSplit"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2],  # Regularization
            'reg_alpha': [0, 0.1, 0.5],  # L1 regularization
            'reg_lambda': [1, 1.5, 2]  # L2 regularization
        }
        
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Calculate sample weights
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        weights = {}
        for cls, count in zip(unique_classes, class_counts):
            weights[cls] = len(y_train) / (len(unique_classes) * count)
        
        sample_weights = np.array([weights[cls] for cls in y_train])
        
        # Grid search
        xgb = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            xgb,
            param_grid,
            cv=tscv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': self.best_params,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores"""
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities


class PolygonXGBoostHFTTrader:
    """Main trading system combining all components"""
    
    def __init__(self, api_key: str, tickers: List[str], initial_capital: float,
                 total_weeks: int = 12, train_weeks: int = 10, test_weeks: int = 2):
        self.data_manager = PolygonDataManager(api_key)
        self.technical_analysis = TechnicalAnalysis()
        self.position_manager = PositionManager(tickers, initial_capital)
        self.models = {}
        self.tickers = tickers
        self.total_weeks = total_weeks
        self.train_weeks = train_weeks
        self.test_weeks = test_weeks
        self.performance_history = []
        
    def initialize_system(self, allocations: Optional[Dict[str, float]] = None):
        """Initialize the trading system"""
        self.position_manager.set_allocations(allocations)
        print(f"Trading System Initialized")
        print(f"   Capital: ${self.position_manager.initial_capital:,.2f}")
        print(f"   Tickers: {', '.join(self.tickers)}")
        print(f"   Allocations: {self.position_manager.allocations}")
    
    def fetch_and_prepare_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch data and calculate indicators for all tickers"""
        prepared_data = {}
        
        for ticker in self.tickers:
            print(f"Processing {ticker}...")
            
            # Fetch raw data
            df = self.data_manager.fetch_ticker_data(ticker, start_date, end_date)
            
            if df.empty:
                print(f"   No data available")
                continue
            
            # Calculate technical indicators
            df = self.technical_analysis.calculate_indicators(df, ticker)
            
            # Prepare features for ML
            if ticker not in self.models:
                self.models[ticker] = XGBoostHFTModel()
            
            df = self.models[ticker].prepare_features(df, ticker)
            
            prepared_data[ticker] = df
            print(f"   {len(df)} data points processed")
        
        return prepared_data
    
    def train_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Train XGBoost models for all tickers"""
        results = {}
        
        for ticker, df in data.items():
            print(f"\nTraining {ticker} model...")
            
            # Clean data
            df_clean = df.dropna()
            
            if len(df_clean) < 1000:
                print(f"   Insufficient data: {len(df_clean)} rows")
                continue
            
            # Split data
            total_rows = len(df_clean)
            test_size = int(total_rows * (self.test_weeks / self.total_weeks))
            train_val_size = total_rows - test_size
            
            # Map signals to labels
            y = df_clean[f'{ticker}_signal'].values  # Already -1, 0, 1
            X = df_clean[self.models[ticker].feature_cols].values
            dates = df_clean['timestamp'].values
            
            # Split
            X_train_val = X[:train_val_size]
            y_train_val = y[:train_val_size]
            X_test = X[train_val_size:]
            y_test = y[train_val_size:]
            dates_test = dates[train_val_size:]
            
            # Grid search
            print(f"   Running GridSearchCV...")
            cv_results = self.models[ticker].grid_search_train(X_train_val, y_train_val)
            print(f"   Best score: {cv_results['best_score']:.4f}")
            
            # Evaluate
            y_pred, y_proba = self.models[ticker].predict_with_confidence(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            results[ticker] = {
                'model': self.models[ticker],
                'accuracy': accuracy,
                'test_dates': (dates_test[0], dates_test[-1]),
                'best_params': cv_results['best_params'],
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            print(f"   Test Accuracy: {accuracy:.4f}")
        
        return results
    
    def execute_trading_cycle(self, current_data: Dict[str, pd.DataFrame]) -> Dict:
        """Execute one complete trading cycle"""
        cycle_results = {}
        current_prices = {}
        
        # Get current prices
        for ticker in self.tickers:
            if ticker in current_data and len(current_data[ticker]) > 0:
                current_prices[ticker] = current_data[ticker]['close'].iloc[-1]
        
        # Check positions and execute trades
        for ticker in self.tickers:
            if ticker not in self.models or ticker not in current_data:
                continue
            
            df = current_data[ticker]
            if df.empty:
                continue
            
            current_price = current_prices[ticker]
            
            # First check stop loss
            if self.position_manager.check_stop_loss(ticker, current_price):
                self.position_manager.execute_sell(ticker, current_price)
                cycle_results[ticker] = 'StopLoss'
                continue
            
            # Get model prediction
            latest_features = df[self.models[ticker].feature_cols].iloc[-1:].values
            prediction, probability = self.models[ticker].predict_with_confidence(latest_features)
            signal = prediction[0]
            
            # Check for pyramid opportunity
            is_pyramid = self.position_manager.should_pyramid_buy(ticker, current_price, signal)
            
            # Execute trades based on signal
            if signal == 1 or is_pyramid:  # Buy signal or pyramid
                if self.position_manager.execute_buy(ticker, current_price, is_pyramid):
                    cycle_results[ticker] = 'Pyramid' if is_pyramid else 'Buy'
            elif signal == -1:  # Sell signal
                if self.position_manager.execute_sell(ticker, current_price):
                    cycle_results[ticker] = 'Sell'
                    # Adapt thresholds based on performance
                    if ticker in cycle_results:
                        performance = 'profitable' if self.position_manager.current_capital > self.position_manager.initial_capital else 'loss'
                        self.models[ticker].adapt_thresholds(performance)
            else:
                cycle_results[ticker] = 'Hold'
        
        # Get portfolio status
        portfolio_status = self.position_manager.get_portfolio_status(current_prices)
        cycle_results['portfolio'] = portfolio_status
        
        return cycle_results
    
    def run_weekly_update(self) -> Dict:
        """Weekly update: roll window, retrain, execute trades"""
        print("\n" + "="*60)
        print("WEEKLY UPDATE")
        print("="*60)
        
        # Fetch new 12-week window
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=self.total_weeks)
        
        # Prepare data
        new_data = self.fetch_and_prepare_data(start_date, end_date)
        
        # Retrain models
        training_results = self.train_models(new_data)
        
        # Execute trading cycle
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
            'volatility': np.std(values) if len(values) > 1 else 0,
            'sharpe_ratio': self.calculate_sharpe_ratio(values),
            'win_rate': self.calculate_win_rate(),
            'trades_executed': sum(1 for record in self.performance_history 
                                 if any(v in ['Buy', 'Sell', 'Pyramid'] 
                                       for v in record['trading_results'].values()))
        }
    
    def calculate_sharpe_ratio(self, values: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(values) < 2:
            return 0
        
        returns = np.diff(values) / values[:-1]
        excess_returns = returns - (risk_free_rate / 52)  # Weekly
        
        if np.std(excess_returns) == 0:
            return 0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(52)
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate from trading history"""
        trades = []
        for record in self.performance_history:
            for ticker, action in record['trading_results'].items():
                if action == 'Sell':
                    trades.append(ticker)
        
        # This is simplified - in production you'd track P&L per trade
        return 0.0  # Placeholder


def main():
    """Example usage"""
    # Configuration
    API_KEY = "YOUR_POLYGON_API_KEY"
    INITIAL_CAPITAL = 50000
    TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]
    
    # Initialize trader
    trader = PolygonXGBoostHFTTrader(
        api_key=API_KEY,
        tickers=TICKERS,
        initial_capital=INITIAL_CAPITAL,
        total_weeks=12,
        train_weeks=10,
        test_weeks=2
    )
    
    # Set capital allocations (optional - defaults to equal)
    trader.initialize_system()
    
    # Run initial training
    print("\nINITIAL SYSTEM SETUP")
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=12)
    
    initial_data = trader.fetch_and_prepare_data(start_date, end_date)
    training_results = trader.train_models(initial_data)
    
    # Execute first trades
    trading_results = trader.execute_trading_cycle(initial_data)
    
    print("\nInitial Portfolio Status:")
    portfolio = trading_results['portfolio']
    print(f"   Value: ${portfolio['portfolio_value']:,.2f}")
    print(f"   Return: {portfolio['return_pct']:+.2f}%")
    print(f"   Capital Usage: {portfolio['capital_usage']:.1%}")
    
    # Simulate weekly updates
    for week in range(4):
        print(f"\n{'='*60}")
        print(f"Week {week + 1}")
        print('='*60)
        
        results = trader.run_weekly_update()
        
        print("\nPortfolio Update:")
        portfolio = results['portfolio']
        print(f"   Value: ${portfolio['portfolio_value']:,.2f}")
        print(f"   Return: {portfolio['return_pct']:+.2f}%")
        
    # Final summary
    summary = trader.get_performance_summary()
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total Return: {summary['total_return']:+.2f}%")
    print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"Total Trades: {summary['trades_executed']}")


if __name__ == "__main__":
    main()