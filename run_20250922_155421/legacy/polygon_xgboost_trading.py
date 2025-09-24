#!/usr/bin/env python3
"""
Polygon.io XGBoost Trading System with Rolling Window
===================================================

This system:
1. Fetches 12 weeks of data from Polygon.io
2. Uses 10 weeks for train/val, 2 weeks for test
3. Applies technical indicators (MACD, RSI, Bollinger Bands)
4. Uses GridSearchCV with TimeSeriesSplit for hyperparameter optimization
5. Retrains weekly by rolling the window forward
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


class PolygonDataFetcher:
    """Handles all Polygon.io data fetching operations"""
    
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)
    
    def fetch_stock_data(self, ticker: str, start_date: datetime, end_date: datetime, 
                        interval: str = "5", timespan: str = "minute") -> pd.DataFrame:
        """Fetch stock data from Polygon.io"""
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
            
            # Add ticker column
            df['ticker'] = ticker
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()


class TechnicalIndicators:
    """Calculate technical indicators for trading signals"""
    
    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
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
    
    @staticmethod
    def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
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
    
    @staticmethod
    def get_trading_signals(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Generate trading signals from indicators"""
        df = df.copy()
        
        # Calculate indicators
        macd_data = TechnicalIndicators.calculate_macd(df[price_col])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        df['rsi'] = TechnicalIndicators.calculate_rsi(df[price_col])
        
        bb_data = TechnicalIndicators.calculate_bollinger_bands(df[price_col])
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        
        # Generate signals
        # MACD signal
        macd_buy = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        macd_sell = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        df['macd_signal_label'] = 'Hold'
        df.loc[macd_buy, 'macd_signal_label'] = 'Buy'
        df.loc[macd_sell, 'macd_signal_label'] = 'Sell'
        
        # RSI signal
        df['rsi_signal_label'] = 'Hold'
        df.loc[df['rsi'] < 30, 'rsi_signal_label'] = 'Buy'
        df.loc[df['rsi'] > 70, 'rsi_signal_label'] = 'Sell'
        
        # Bollinger Bands signal
        df['bb_signal_label'] = 'Hold'
        df.loc[df[price_col] <= df['bb_lower'], 'bb_signal_label'] = 'Buy'
        df.loc[df[price_col] >= df['bb_upper'], 'bb_signal_label'] = 'Sell'
        
        # Voting system
        def get_voting_signal(row):
            signals = [row['macd_signal_label'], row['rsi_signal_label'], row['bb_signal_label']]
            signal_counts = pd.Series(signals).value_counts()
            
            if len(signal_counts) == 0:
                return 'Hold'
            elif signal_counts.max() == 1:
                return 'Hold'  # No clear majority
            else:
                return signal_counts.idxmax()
        
        df['final_signal'] = df.apply(get_voting_signal, axis=1)
        
        return df


class XGBoostTradingModel:
    """XGBoost model with GridSearchCV for trading"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_cols = None
        self.label_map = {'Sell': 0, 'Hold': 1, 'Buy': 2}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for XGBoost"""
        feature_cols = []
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_momentum'] = df['close'] - df['close'].shift(5)
        feature_cols.extend(['returns', 'log_returns', 'price_momentum'])
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        feature_cols.append('volume_ratio')
        
        # Technical indicator features
        feature_cols.extend(['macd', 'macd_signal', 'macd_histogram', 'rsi'])
        
        # Bollinger Band features
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        feature_cols.extend(['bb_width', 'bb_position'])
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            feature_cols.extend([f'close_lag_{lag}', f'volume_lag_{lag}'])
        
        self.feature_cols = feature_cols
        return df
    
    def grid_search_cv(self, X_train: np.ndarray, y_train: np.ndarray, 
                      cv_splits: int = 5) -> Dict:
        """Perform GridSearchCV with TimeSeriesSplit"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Calculate sample weights based on class imbalance
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        n_samples = len(y_train)
        n_classes = len(unique_classes)
        
        weights = {}
        for cls, count in zip(unique_classes, class_counts):
            weights[cls] = n_samples / (n_classes * count)
        
        sample_weights = np.array([weights[cls] for cls in y_train])
        
        # Grid search
        xgb = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            tree_method='hist'
        )
        
        grid_search = GridSearchCV(
            xgb,
            param_grid,
            cv=tscv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit with sample weights
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': self.best_params,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train the model with best parameters"""
        # Calculate sample weights
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        n_samples = len(y_train)
        n_classes = len(unique_classes)
        
        weights = {}
        for cls, count in zip(unique_classes, class_counts):
            weights[cls] = n_samples / (n_classes * count)
        
        sample_weights = np.array([weights[cls] for cls in y_train])
        
        # Train with best parameters
        self.model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            tree_method='hist',
            **self.best_params
        )
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return self.model.predict_proba(X)


class RollingWindowTrader:
    """Main trading system with rolling window training"""
    
    def __init__(self, api_key: str, tickers: List[str], 
                 total_weeks: int = 12, train_weeks: int = 10, test_weeks: int = 2):
        self.data_fetcher = PolygonDataFetcher(api_key)
        self.tickers = tickers
        self.total_weeks = total_weeks
        self.train_weeks = train_weeks
        self.test_weeks = test_weeks
        self.models = {}
        self.performance_history = {}
    
    def fetch_initial_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch initial 12 weeks of data for all tickers"""
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=self.total_weeks)
        
        data = {}
        for ticker in self.tickers:
            print(f"Fetching data for {ticker}...")
            df = self.data_fetcher.fetch_stock_data(ticker, start_date, end_date)
            
            if not df.empty:
                # Add technical indicators
                df = TechnicalIndicators.get_trading_signals(df)
                data[ticker] = df
            else:
                print(f"No data available for {ticker}")
        
        return data
    
    def train_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Train models for all tickers"""
        results = {}
        
        for ticker, df in data.items():
            print(f"\nTraining model for {ticker}...")
            
            # Prepare features
            model = XGBoostTradingModel()
            df = model.prepare_features(df)
            
            # Remove NaN values
            df_clean = df.dropna()
            
            if len(df_clean) < 1000:  # Need sufficient data
                print(f"Insufficient data for {ticker}: {len(df_clean)} rows")
                continue
            
            # Split data: last 2 weeks for test, rest for train/val
            total_rows = len(df_clean)
            test_size = int(total_rows * (self.test_weeks / self.total_weeks))
            train_val_size = total_rows - test_size
            
            # Prepare features and labels
            X = df_clean[model.feature_cols].values
            y = df_clean['final_signal'].map(model.label_map).values
            dates = df_clean['timestamp'].values
            
            # Split data
            X_train_val = X[:train_val_size]
            y_train_val = y[:train_val_size]
            X_test = X[train_val_size:]
            y_test = y[train_val_size:]
            dates_test = dates[train_val_size:]
            
            # Grid search on train/val set
            print(f"Running GridSearchCV for {ticker}...")
            cv_results = model.grid_search_cv(X_train_val, y_train_val, cv_splits=5)
            print(f"Best params for {ticker}: {cv_results['best_params']}")
            print(f"Best CV score: {cv_results['best_score']:.4f}")
            
            # Train final model on all train/val data
            model.train(X_train_val, y_train_val)
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Generate classification report
            report = classification_report(
                y_test, y_pred,
                target_names=['Sell', 'Hold', 'Buy'],
                output_dict=True
            )
            
            # Calculate weekly recommendations
            majority_vote = pd.Series(y_pred).value_counts().idxmax()
            avg_proba = y_proba.mean(axis=0)
            
            results[ticker] = {
                'model': model,
                'accuracy': accuracy,
                'classification_report': report,
                'majority_vote': model.inv_label_map[majority_vote],
                'avg_probabilities': {
                    'Sell': avg_proba[0],
                    'Hold': avg_proba[1],
                    'Buy': avg_proba[2]
                },
                'test_dates': (dates_test[0], dates_test[-1]),
                'best_params': cv_results['best_params']
            }
            
            # Store model
            self.models[ticker] = model
            
            print(f"{ticker} Test Accuracy: {accuracy:.4f}")
            print(f"{ticker} Weekly Recommendation: {results[ticker]['majority_vote']}")
        
        return results
    
    def update_weekly(self) -> Dict[str, Dict]:
        """Remove oldest week, add newest week, and retrain"""
        print("\n=== Weekly Update ===")
        
        # Fetch new week of data
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=1)
        
        updated_results = {}
        
        for ticker in self.tickers:
            if ticker not in self.models:
                continue
            
            print(f"\nUpdating {ticker}...")
            
            # Fetch new data
            new_data = self.data_fetcher.fetch_stock_data(ticker, start_date, end_date)
            
            if new_data.empty:
                print(f"No new data for {ticker}")
                continue
            
            # Process new data with indicators
            new_data = TechnicalIndicators.get_trading_signals(new_data)
            
            # Fetch full data range (shift window by 1 week)
            full_end = datetime.now()
            full_start = full_end - timedelta(weeks=self.total_weeks)
            
            full_data = self.data_fetcher.fetch_stock_data(ticker, full_start, full_end)
            
            if full_data.empty:
                continue
            
            # Process and retrain
            full_data = TechnicalIndicators.get_trading_signals(full_data)
            
            # Retrain model with new data
            model_data = {ticker: full_data}
            new_results = self.train_models(model_data)
            
            if ticker in new_results:
                updated_results[ticker] = new_results[ticker]
                
                # Store performance history
                if ticker not in self.performance_history:
                    self.performance_history[ticker] = []
                
                self.performance_history[ticker].append({
                    'date': datetime.now(),
                    'accuracy': new_results[ticker]['accuracy'],
                    'recommendation': new_results[ticker]['majority_vote']
                })
        
        return updated_results
    
    def get_current_recommendations(self) -> Dict[str, Dict]:
        """Get current trading recommendations for all tickers"""
        recommendations = {}
        
        for ticker, model in self.models.items():
            # Fetch latest data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            latest_data = self.data_fetcher.fetch_stock_data(ticker, start_date, end_date)
            
            if latest_data.empty:
                continue
            
            # Process with indicators
            latest_data = TechnicalIndicators.get_trading_signals(latest_data)
            latest_data = model.prepare_features(latest_data)
            
            # Get latest valid row
            latest_clean = latest_data.dropna()
            
            if len(latest_clean) == 0:
                continue
            
            # Predict
            X_latest = latest_clean[model.feature_cols].iloc[-1:].values
            pred = model.predict(X_latest)[0]
            proba = model.predict_proba(X_latest)[0]
            
            recommendations[ticker] = {
                'signal': model.inv_label_map[pred],
                'probabilities': {
                    'Sell': proba[0],
                    'Hold': proba[1],
                    'Buy': proba[2]
                },
                'timestamp': latest_clean['timestamp'].iloc[-1]
            }
        
        return recommendations


def main():
    """Example usage of the trading system"""
    
    # Configuration
    API_KEY = "YOUR_POLYGON_API_KEY"  # Replace with your actual API key
    TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]
    
    # Initialize trading system
    trader = RollingWindowTrader(
        api_key=API_KEY,
        tickers=TICKERS,
        total_weeks=12,
        train_weeks=10,
        test_weeks=2
    )
    
    # Initial training
    print("=== Initial Data Fetch and Training ===")
    initial_data = trader.fetch_initial_data()
    initial_results = trader.train_models(initial_data)
    
    # Display initial results
    print("\n=== Initial Training Results ===")
    for ticker, results in initial_results.items():
        print(f"\n{ticker}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Weekly Recommendation: {results['majority_vote']}")
        print(f"  Probabilities: Sell={results['avg_probabilities']['Sell']:.3f}, "
              f"Hold={results['avg_probabilities']['Hold']:.3f}, "
              f"Buy={results['avg_probabilities']['Buy']:.3f}")
    
    # Simulate weekly updates
    for week in range(4):  # Run for 4 weeks
        print(f"\n{'='*50}")
        print(f"Week {week + 1} Update")
        print('='*50)
        
        # Wait for a week (in production)
        # In simulation, we can just proceed
        time.sleep(1)  # Small delay for demonstration
        
        # Update models
        weekly_results = trader.update_weekly()
        
        # Display weekly results
        print("\n=== Weekly Update Results ===")
        for ticker, results in weekly_results.items():
            print(f"\n{ticker}:")
            print(f"  New Accuracy: {results['accuracy']:.4f}")
            print(f"  New Recommendation: {results['majority_vote']}")
        
        # Get current recommendations
        current_recs = trader.get_current_recommendations()
        
        print("\n=== Current Trading Signals ===")
        for ticker, rec in current_recs.items():
            print(f"{ticker}: {rec['signal']} (as of {rec['timestamp']})")
    
    # Display performance history
    print("\n=== Performance History ===")
    for ticker, history in trader.performance_history.items():
        print(f"\n{ticker}:")
        for record in history:
            print(f"  {record['date'].strftime('%Y-%m-%d')}: "
                  f"Accuracy={record['accuracy']:.4f}, "
                  f"Recommendation={record['recommendation']}")


if __name__ == "__main__":
    main()