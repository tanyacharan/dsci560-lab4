import pandas as pd
import numpy as np
from typing import Optional
import time
import random
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score

class AlpacaConstraints:
    """Constraints and limitations of selected trading platform Alpaca."""
    # Probably should be in another file. TODO: Note later
    COMMISSION_PER_SHARE = 0.001 # Under 10M shares/month
    COMMISSION_PER_SHARE_10M = 0.0005 # Over 10M shares/month - not happening
    FINRA_TAF_FEE = 0.000166 # $0.000166/share
    SEC_FEE = 0.0000278 # $0.0000278/share
    # No minimum commission
    
    # Position constraints  
    MIN_FRACTIONAL_DOLLARS = 1.00
    MIN_FRACTIONAL_SHARES = 0.001
    MAX_SHARES_PER_ORDER = 10000
    
    # Margin
    MARGIN_RATE_ANNUAL = 0.07  # 7%
    INTRADAY_LEVERAGE = 4.0
    OVERNIGHT_LEVERAGE = 2.0
    
    # PDT Rules
    PDT_MIN_EQUITY = 25000
    PDT_WEEKLY_LIMIT = 3  # If under $25k

class StockPositions:
    """Create buy-in and sell-out positions based on stock information."""

    def __init__(self, tickers: list[str], init_cap: int) -> None:
        self.tickers = tickers
        # (ticker, max buy-in)
        self.buyin_portfolio = dict(zip(tickers, [0] * len(tickers)))
        # (ticker, current number of stocks owned) - we assume no previous ownership
        self.current_own = dict(zip(tickers, [0] * len(tickers)))
        # (ticker, most recent buy price) - 0 if no position
        self.recent_buy_prices = dict(zip(tickers, [0.0] * len(tickers)))
        # Stop loss in basis points (3-5 bps for HFT)
        self.stop_loss_bps = 4  # 4 basis points = 0.04%
        self.init_cap = init_cap
        self.current_cap = init_cap
        self.portfolio_value = init_cap
        # This is probably not optimal but whatever - TODO: consider?
        self.platform = AlpacaConstraints

    def set_buyin(self, tickers = None) -> None:
        # If the user wants to change certain tickers
        if tickers is not None:
            for ticker in self.tickers:
                buyin = int(input(f"Input your buy-in capital for ticker {ticker}:"))
                self.buyin_portfolio[ticker] = buyin
        # Should be called during creation
        else:
            for ticker in self.tickers:
                buyin = int(input(f"Input your buy-in capital for ticker {ticker}:"))
                self.buyin_portfolio[ticker] = buyin
        print("Buy-in capital setup success!")
        
    def set_buyin_programmatically(self, allocations: dict) -> None:
        """Set buy-in allocations programmatically without user input."""
        for ticker, amount in allocations.items():
            if ticker in self.buyin_portfolio:
                self.buyin_portfolio[ticker] = amount
        print("Buy-in capital setup complete!")
        
    def print_buyin(self) -> None:
        print("Buy-in capital of current stocks:")
        for ticker, price in self.buyin_portfolio.items():
            print(f"Ticker {ticker} buy-in: ${price}")

    def _stock_price_check(self, ticker: str) -> float:
        # For now, return a simulated price based on some basic logic
        # This will be replaced with your "fake" data injection system
        base_price = 100 + hash(ticker) % 200  # Deterministic base price per ticker
        volatility = random.uniform(0.95, 1.05)  # 5% volatility
        return round(base_price * volatility, 2)
    
    def inject_price_data(self, ticker: str, timestamp, price: float, volume: float = 0) -> None:
        """Method to inject 'fake' price data for testing."""
        # Get or create predictor for this ticker
        if not hasattr(self, 'predictors'):
            self.predictors = {}
        
        if ticker not in self.predictors:
            self.predictors[ticker] = TradingPredictors(ticker)
        
        # Update predictor with the injected price data
        self.predictors[ticker].add_price_data(timestamp, price)
    
    def get_current_signals(self, ticker: str) -> dict:
        """Get current trading signals and predictions for a ticker."""
        if not hasattr(self, 'predictors') or ticker not in self.predictors:
            return {"error": f"No predictor found for {ticker}"}
        
        predictor = self.predictors[ticker]
        
        # Get all current analysis
        signals = predictor._get_indicator_signals()
        vote_result = predictor._voting_system(signals)
        bp_change = predictor._calculate_basis_points_change()
        
        result = {
            "ticker": ticker,
            "individual_signals": signals,
            "voting_result": vote_result,
            "bp_change": bp_change,
            "bp_threshold": predictor.bp_threshold,
            "model_trained": predictor.model_trained,
            "buy_conditions": predictor._buy_conditions(ticker),
            "sell_conditions": predictor._sell_conditions(ticker)
        }
        
        # Add ML predictions if available
        if predictor.model_trained:
            result["ml_predictions"] = {
                "predicted_bp": predictor._predict_bp_change(),
                "predicted_signal": predictor._predict_signal()
            }
        
        return result
    
    def _portfolio_value_check(self) -> float:
        """Calculate total portfolio value (cash + stock positions)."""
        stock_value = 0.0
        for ticker in self.tickers:
            if self.current_own[ticker] > 0:
                stock_price = self._stock_price_check(ticker)
                stock_value += self.current_own[ticker] * stock_price
        
        self.portfolio_value = self.current_cap + stock_value
        return self.portfolio_value
    
    def _transaction_costs_buy(self, count: int) -> float:
        comms = self.platform.COMMISSION_PER_SHARE * count
        return comms

    def _action_buy(self, ticker: str) -> None:
        stock_price = self._stock_price_check(ticker)
        
        # Determine buy size - smaller for pyramid buys
        is_new_position = self.current_own[ticker] == 0
        if is_new_position:
            # Initial position: use full allocation
            max_spend = min(self.buyin_portfolio[ticker], self.current_cap)
        else:
            # Pyramid buy: use smaller allocation (25% of remaining capital)
            max_spend = min(self.current_cap * 0.25, self.buyin_portfolio[ticker] * 0.5)
        
        count = int(max_spend / stock_price)
        if count == 0:
            print(f"Cannot proceed with purchase - not enough funds. Available: ${self.current_cap:.2f}")
            return
        
        # Calculate total cost
        cost = round(count * stock_price, 4) + self._transaction_costs_buy(count)
        
        # Execute purchase
        self.current_own[ticker] += count
        self.current_cap -= cost
        
        # Update predictor position tracking
        if hasattr(self, 'predictors') and ticker in self.predictors:
            self.predictors[ticker].add_position_level(stock_price, count)
        
        # Save the buy price
        self.recent_buy_prices[ticker] = stock_price
        
        # Print action
        action_type = "Initial buy" if is_new_position else "Pyramid buy"
        print(f"{action_type} {ticker}: {count} shares at ${stock_price:.2f} (Total: {self.current_own[ticker]} shares)")
        
        # Warning system post-buy
        if self.current_cap <= 0.1 * self.init_cap:
            print("Warning: Low funds.")
        return

    def _transaction_costs_sell(self, count: int) -> float:
        comms = self.platform.COMMISSION_PER_SHARE * count
        finra_taf = self.platform.FINRA_TAF_FEE * count
        sec_fee = self.platform.SEC_FEE * count
        return comms + finra_taf + sec_fee
        
    def _action_sell(self, ticker: str) -> None:
        stock_price = self._stock_price_check(ticker)
        # We assume we will sell all stocks.
        count = self.current_own[ticker]
        if count == 0:
            print(f"No position to sell for {ticker}")
            return
            
        # Calculate total earnings
        earnings = round(count * stock_price, 4) - self._transaction_costs_sell(count)
        
        # Calculate profit/loss for threshold adaptation
        if self.recent_buy_prices[ticker] > 0:
            total_cost = count * self.recent_buy_prices[ticker]  # Simplified - doesn't account for pyramid
            profit_loss = earnings - total_cost
            performance = "profitable" if profit_loss > 0 else "loss"
            
            # Update predictor thresholds based on performance
            if hasattr(self, 'predictors') and ticker in self.predictors:
                self.predictors[ticker]._adapt_bp_threshold(performance)
                self.predictors[ticker].clear_position_levels()
        
        # Execute sale
        self.current_own[ticker] = 0
        self.current_cap += earnings
        
        # Reset buy price when position is closed
        self.recent_buy_prices[ticker] = 0.0
        
        print(f"Sold all {count} shares of {ticker} at ${stock_price:.2f} for ${earnings:.2f}")
        
    def _check_stop_loss(self, ticker: str, current_price: float) -> bool:
        """Check if stop loss is triggered based on HFT parameters."""
        if self.current_own[ticker] == 0 or self.recent_buy_prices[ticker] == 0:
            return False
        
        buy_price = self.recent_buy_prices[ticker]
        # Calculate price drop in basis points
        price_drop_bps = ((buy_price - current_price) / buy_price) * 10000
        
        if price_drop_bps >= self.stop_loss_bps:
            print(f"Stop loss triggered for {ticker}: {price_drop_bps:.1f} bps drop")
            return True
        return False
    
    def _action_hold(self, ticker: str) -> None:
        if self.current_own[ticker] > 0:
            print(f"Holding {self.current_own[ticker]} of stock {ticker}.")

    def trade(self) -> None:
        """Trade stocks if sufficient requirements are made."""
        # TODO: Figure this out
        for ticker in self.tickers:
            # First check stop loss for any existing positions
            current_price = self._stock_price_check(ticker)
            if self._check_stop_loss(ticker, current_price):
                self._action_sell(ticker)
                continue  # Skip to next ticker since we just sold
                
            # Get or create predictor for this ticker
            if not hasattr(self, 'predictors'):
                self.predictors = {}
            
            if ticker not in self.predictors:
                self.predictors[ticker] = TradingPredictors(ticker)
            
            # Update predictor with current price data
            current_price = self._stock_price_check(ticker)
            current_time = time.time()
            self.predictors[ticker].add_price_data(current_time, current_price)
            
            # Make trading decision
            ticker_predictions = self.predictors[ticker]
            if ticker_predictions._buy_conditions(ticker):
                self._action_buy(ticker)
            elif ticker_predictions._sell_conditions(ticker):
                self._action_sell(ticker)
            else:
                self._action_hold(ticker)
        
        # Update portfolio value
        self._portfolio_value_check()
        print(f"Current capital: {self.current_cap}")
        print(f"Current stock holdings:")
        for ticker in self.tickers:
            print(self.current_own[ticker])
    
    # TODO: I guess we're not doing margin trading for now, but note that for later.
    # TODO: We need to record buy price and set a lower bound limit to which we must sell.

class TradingPredictors:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.price_history = []  # Will store (timestamp, price) tuples
        self.features = {}  # Store calculated indicators
        self.bp_threshold_buy = 10  # Start with 10 basis points for buy signal
        self.bp_threshold_sell = 4   # Start with 4 basis points for sell signal
        self.bp_threshold_max = 30   # Maximum threshold
        self.bp_threshold_increment = 1.0  # How much to increase threshold
        
        # Pyramid trading variables
        self.position_levels = []  # Track buy-in levels [(price, quantity)]
        self.last_buy_price = 0.0
        self.pyramid_increment = 5  # BP increase needed for next pyramid buy
        
        # XGBoost models (will be initialized when we have enough data)
        self.regression_model = None
        self.classification_model = None
        self.model_trained = False
        self.min_training_samples = 50  # Minimum samples needed to train models
        
        # Rolling weekly training data storage
        self.weekly_training_data = []  # List of weekly batches
        self.max_weeks_history = 8  # Keep last 8 weeks for training
        self.current_week_data = []  # Current week's data being collected
        
        # Hyperparameter optimization settings
        self.optimize_hyperparams = True
        self.best_reg_params = None
        self.best_clf_params = None
    
    def add_price_data(self, timestamp, price: float) -> None:
        """Add new price data point for indicator calculations."""
        self.price_history.append((timestamp, price))
        # Keep only last 200 periods for efficiency
        if len(self.price_history) > 200:
            self.price_history = self.price_history[-200:]
    
    def _get_price_series(self) -> list[float]:
        """Extract price values from history."""
        return [price for _, price in self.price_history]
    
    def _calculate_macd(self, fast=12, slow=26, signal=9) -> dict:
        """Calculate MACD indicator."""
        prices = self._get_price_series()
        if len(prices) < slow + signal:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        price_series = pd.Series(prices)
        
        # Calculate EMAs
        ema_fast = price_series.ewm(span=fast, adjust=False).mean()
        ema_slow = price_series.ewm(span=slow, adjust=False).mean()
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        # Histogram
        histogram = macd - signal_line
        
        return {
            "macd": float(macd.iloc[-1]) if len(macd) > 0 else 0,
            "signal": float(signal_line.iloc[-1]) if len(signal_line) > 0 else 0,
            "histogram": float(histogram.iloc[-1]) if len(histogram) > 0 else 0
        }
    
    def _calculate_rsi(self, window=14) -> float:
        """Calculate RSI indicator."""
        prices = self._get_price_series()
        if len(prices) < window + 1:
            return 50  # Neutral RSI
        
        price_series = pd.Series(prices)
        
        # Calculate price changes
        delta = price_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gains and losses
        avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
        
        # Relative strength and RSI
        rs = avg_gain / avg_loss.replace(0, float('inf'))
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if len(rsi) > 0 else 50
    
    def _calculate_bollinger_bands(self, window=20, num_std=2) -> dict:
        """Calculate Bollinger Bands."""
        prices = self._get_price_series()
        if len(prices) < window:
            current_price = prices[-1] if prices else 100
            return {
                "upper": current_price * 1.02,
                "middle": current_price,
                "lower": current_price * 0.98
            }
        
        price_series = pd.Series(prices)
        
        # Calculate rolling mean and standard deviation
        rolling_mean = price_series.rolling(window=window).mean()
        rolling_std = price_series.rolling(window=window).std()
        
        # Calculate bands
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return {
            "upper": float(upper_band.iloc[-1]) if len(upper_band) > 0 else 0,
            "middle": float(rolling_mean.iloc[-1]) if len(rolling_mean) > 0 else 0,
            "lower": float(lower_band.iloc[-1]) if len(lower_band) > 0 else 0
        }
    
    def _update_features(self) -> None:
        """Update all technical indicators."""
        if len(self.price_history) < 2:
            return
        
        self.features.update({
            "macd": self._calculate_macd(),
            "rsi": self._calculate_rsi(),
            "bollinger": self._calculate_bollinger_bands()
        })
    
    def _get_indicator_signals(self) -> dict:
        """Generate buy/sell signals from individual indicators."""
        self._update_features()
        
        signals = {"macd": "Hold", "rsi": "Hold", "bollinger": "Hold"}
        
        if not self.features:
            return signals
        
        # MACD signals
        macd_data = self.features.get("macd", {})
        if macd_data.get("macd", 0) > macd_data.get("signal", 0):
            signals["macd"] = "Buy"
        elif macd_data.get("macd", 0) < macd_data.get("signal", 0):
            signals["macd"] = "Sell"
        
        # RSI signals
        rsi = self.features.get("rsi", 50)
        if rsi < 30:
            signals["rsi"] = "Buy"  # Oversold
        elif rsi > 70:
            signals["rsi"] = "Sell"  # Overbought
        
        # Bollinger Bands signals
        bb = self.features.get("bollinger", {})
        current_price = self._get_price_series()[-1] if self.price_history else 0
        if current_price and bb.get("lower", 0):
            if current_price <= bb["lower"]:
                signals["bollinger"] = "Buy"
            elif current_price >= bb["upper"]:
                signals["bollinger"] = "Sell"
        
        return signals
    
    def _voting_system(self, signals: dict) -> str:
        """Determine final signal based on majority vote."""
        votes = list(signals.values())
        
        # Count votes
        buy_votes = votes.count("Buy")
        sell_votes = votes.count("Sell")
        hold_votes = votes.count("Hold")
        
        # Majority wins, ties default to Hold
        if buy_votes > sell_votes and buy_votes > hold_votes:
            return "Buy"
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            return "Sell"
        else:
            return "Hold"
    
    def _calculate_basis_points_change(self) -> float:
        """Calculate recent basis points change for regression model."""
        if len(self.price_history) < 2:
            return 0.0
        
        recent_price = self.price_history[-1][1]
        previous_price = self.price_history[-2][1]
        
        # Calculate basis points change
        bp_change = ((recent_price - previous_price) / previous_price) * 10000
        return bp_change
    
    def _adapt_bp_threshold(self, recent_performance: str) -> None:
        """Adapt basis point thresholds based on recent performance."""
        if recent_performance == "profitable":
            # If profitable, raise the bar for both buy and sell
            self.bp_threshold_buy = min(self.bp_threshold_buy + self.bp_threshold_increment, 
                                      self.bp_threshold_max)
            self.bp_threshold_sell = min(self.bp_threshold_sell + self.bp_threshold_increment, 
                                       self.bp_threshold_max)
        elif recent_performance == "loss":
            # If losing, be less selective (decrease thresholds)
            self.bp_threshold_buy = max(self.bp_threshold_buy - self.bp_threshold_increment, 5.0)
            self.bp_threshold_sell = max(self.bp_threshold_sell - self.bp_threshold_increment, 2.0)
    
    def _should_pyramid_buy(self, current_price: float, predicted_bp: float) -> bool:
        """Determine if we should add to existing position (pyramid buying)."""
        if not self.position_levels or self.last_buy_price == 0:
            return False
        
        # Calculate BP increase since last buy
        bp_since_last_buy = ((current_price - self.last_buy_price) / self.last_buy_price) * 10000
        
        # Pyramid if price has risen enough AND we predict more upside
        return (bp_since_last_buy >= self.pyramid_increment and 
                predicted_bp >= self.bp_threshold_buy)
    
    def add_position_level(self, price: float, quantity: int) -> None:
        """Track position levels for pyramid strategy."""
        self.position_levels.append((price, quantity))
        self.last_buy_price = price
    
    def clear_position_levels(self) -> None:
        """Clear position tracking when fully sold."""
        self.position_levels = []
        self.last_buy_price = 0.0
    
    def _prepare_features(self) -> Optional[np.ndarray]:
        """Prepare feature vector for ML models."""
        if not self.features or len(self.price_history) < 5:
            return None
        
        # Extract features for ML model
        feature_vector = []
        
        # MACD features
        macd_data = self.features.get("macd", {})
        feature_vector.extend([
            macd_data.get("macd", 0),
            macd_data.get("signal", 0),
            macd_data.get("histogram", 0)
        ])
        
        # RSI feature
        feature_vector.append(self.features.get("rsi", 50))
        
        # Bollinger Bands features
        bb_data = self.features.get("bollinger", {})
        current_price = self._get_price_series()[-1] if self.price_history else 0
        feature_vector.extend([
            bb_data.get("upper", current_price * 1.02),
            bb_data.get("middle", current_price),
            bb_data.get("lower", current_price * 0.98),
            current_price  # Current price itself
        ])
        
        # Price momentum features (last 3 price changes)
        prices = self._get_price_series()
        if len(prices) >= 4:
            for i in range(3):
                if len(prices) > i + 1:
                    change = (prices[-(i+1)] - prices[-(i+2)]) / prices[-(i+2)] * 10000  # BP change
                    feature_vector.append(change)
                else:
                    feature_vector.append(0)
        else:
            feature_vector.extend([0, 0, 0])
        
        return np.array(feature_vector).reshape(1, -1)
    
    def _collect_training_data(self) -> None:
        """Collect training data for current week."""
        features = self._prepare_features()
        if features is None:
            return
        
        # Calculate target for regression (next period BP change)
        bp_change = self._calculate_basis_points_change()
        
        # Calculate target for classification (indicator vote)
        signals = self._get_indicator_signals()
        vote_result = self._voting_system(signals)
        
        # Add to current week's data
        data_point = {
            'features': features.flatten(),
            'bp_change': bp_change,
            'signal': vote_result,
            'timestamp': time.time()
        }
        self.current_week_data.append(data_point)
    
    def finalize_week_and_train(self) -> bool:
        """Finalize current week's data and retrain models."""
        if len(self.current_week_data) < 10:  # Need minimum data per week
            print(f"Insufficient data for {self.ticker} this week: {len(self.current_week_data)} samples")
            self.current_week_data = []  # Reset anyway
            return False
        
        # Add current week to history
        self.weekly_training_data.append(self.current_week_data.copy())
        
        # Maintain rolling window - remove oldest week if we exceed max
        if len(self.weekly_training_data) > self.max_weeks_history:
            self.weekly_training_data = self.weekly_training_data[-self.max_weeks_history:]
        
        # Train on all available weeks
        success = self._train_on_historical_weeks()
        
        # Reset current week
        self.current_week_data = []
        
        return success
    
    def _train_on_historical_weeks(self) -> bool:
        """Train models on all historical weekly data."""
        if len(self.weekly_training_data) == 0:
            return False
        
        # Combine all weeks into training arrays
        all_features = []
        all_bp_changes = []
        all_signals = []
        
        for week_data in self.weekly_training_data:
            for data_point in week_data:
                all_features.append(data_point['features'])
                all_bp_changes.append(data_point['bp_change'])
                all_signals.append(data_point['signal'])
        
        if len(all_features) < self.min_training_samples:
            print(f"Not enough combined data for {self.ticker}: {len(all_features)} samples")
            return False
        
        try:
            X = np.array(all_features)
            y_reg = np.array(all_bp_changes)
            y_clf = np.array(all_signals)
            
            # Optimize hyperparameters if enabled and we have enough data
            if self.optimize_hyperparams and len(X) >= 200:
                reg_params, clf_params = self._optimize_hyperparameters(X, y_reg, y_clf)
            else:
                # Use default params or previously optimized ones
                reg_params = self.best_reg_params or {
                    'n_estimators': 100,
                    'max_depth': 4,
                    'learning_rate': 0.1
                }
                clf_params = self.best_clf_params or {
                    'n_estimators': 100,
                    'max_depth': 4,
                    'learning_rate': 0.1
                }
            
            # Train ticker-specific regression model
            self.regression_model = XGBRegressor(
                random_state=42 + hash(self.ticker) % 100,
                **reg_params
            )
            self.regression_model.fit(X, y_reg)
            
            # Train ticker-specific classification model
            label_map = {"Sell": 0, "Hold": 1, "Buy": 2}
            y_clf_numeric = [label_map.get(label, 1) for label in y_clf]
            
            self.classification_model = XGBClassifier(
                random_state=42 + hash(self.ticker) % 100,
                **clf_params
            )
            self.classification_model.fit(X, y_clf_numeric)
            
            self.model_trained = True
            weeks_used = len(self.weekly_training_data)
            print(f"{self.ticker} models retrained on {weeks_used} weeks ({len(X)} total samples)")
            return True
            
        except Exception as e:
            print(f"Error training {self.ticker} models: {e}")
            return False
    
    def _optimize_hyperparameters(self, X, y_reg, y_clf):
        """Optimize hyperparameters using TimeSeriesSplit cross-validation."""
        print(f"Optimizing hyperparameters for {self.ticker}...")
        
        # Parameter grids
        reg_param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        clf_param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 4, 6],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Use TimeSeriesSplit for time-aware cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        try:
            # Optimize regression model
            reg_base = XGBRegressor(random_state=42 + hash(self.ticker) % 100)
            reg_grid = GridSearchCV(
                reg_base, reg_param_grid, 
                cv=tscv, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=0
            )
            reg_grid.fit(X, y_reg)
            self.best_reg_params = reg_grid.best_params_
            
            # Optimize classification model
            label_map = {"Sell": 0, "Hold": 1, "Buy": 2}
            y_clf_numeric = [label_map.get(label, 1) for label in y_clf]
            
            clf_base = XGBClassifier(random_state=42 + hash(self.ticker) % 100)
            clf_grid = GridSearchCV(
                clf_base, clf_param_grid,
                cv=tscv, scoring='accuracy',
                n_jobs=-1, verbose=0
            )
            clf_grid.fit(X, y_clf_numeric)
            self.best_clf_params = clf_grid.best_params_
            
            print(f"{self.ticker} hyperparameter optimization complete")
            return self.best_reg_params, self.best_clf_params
            
        except Exception as e:
            print(f"Hyperparameter optimization failed for {self.ticker}: {e}")
            # Return default params
            default_reg = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.9}
            default_clf = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.9}
            return default_reg, default_clf
    
    def _train_models(self) -> None:
        """Train XGBoost models if enough data is available."""
        try:
            X = np.array(self.training_features)
            y_reg = np.array(self.training_targets_regression)
            y_clf = np.array(self.training_targets_classification)
            
            # Train regression model for BP prediction
            self.regression_model = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.regression_model.fit(X, y_reg)
            
            # Train classification model for signal prediction
            # Convert string labels to numeric
            label_map = {"Sell": 0, "Hold": 1, "Buy": 2}
            y_clf_numeric = [label_map.get(label, 1) for label in y_clf]
            
            self.classification_model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.classification_model.fit(X, y_clf_numeric)
            
            self.model_trained = True
            print(f"Models trained for {self.ticker} with {len(X)} samples")
            
        except Exception as e:
            print(f"Error training models for {self.ticker}: {e}")
    
    def _predict_next_price(self) -> float:
        """Predict next period's price using regression model."""
        if not self.model_trained or self.regression_model is None:
            return 0.0
        
        features = self._prepare_features()
        if features is None:
            return 0.0
        
        try:
            prediction = self.regression_model.predict(features)[0]
            return float(prediction)
        except Exception as e:
            print(f"Error predicting price for {self.ticker}: {e}")
            return 0.0
    
    def _predict_signal(self) -> str:
        """Predict trading signal using classification model."""
        if not self.model_trained or self.classification_model is None:
            return "Hold"
        
        features = self._prepare_features()
        if features is None:
            return "Hold"
        
        try:
            prediction = self.classification_model.predict(features)[0]
            label_map = {0: "Sell", 1: "Hold", 2: "Buy"}
            return label_map.get(prediction, "Hold")
        except Exception as e:
            print(f"Error predicting signal for {self.ticker}: {e}")
            return "Hold"
    
    def _buy_conditions(self, ticker: str) -> bool:
        """Determine if buy conditions are met using ensemble of methods."""
        if len(self.price_history) < 10:  # Need some history
            return False
        
        # Collect training data
        self._collect_training_data()
        
        # Train models periodically
        if len(self.training_features) >= self.min_training_samples and len(self.training_features) % 20 == 0:
            self._train_models()
        
        # Get indicator signals (voting system)
        signals = self._get_indicator_signals()
        vote_result = self._voting_system(signals)
        
        # Check basis points change
        bp_change = self._calculate_basis_points_change()
        
        # Get ML predictions if available
        predicted_price = self._predict_next_price() if self.model_trained else 0
        predicted_signal = self._predict_signal() if self.model_trained else "Hold"
        
        # Calculate predicted price change in BP
        current_price = self._get_price_series()[-1] if self.price_history else 0
        predicted_bp = 0
        if current_price > 0 and predicted_price > 0:
            predicted_bp = ((predicted_price - current_price) / current_price) * 10000
        
        # Ensemble decision logic
        buy_signals = 0
        
        # Traditional indicator vote
        if vote_result == "Buy":
            buy_signals += 1
        
        # Current BP change meets threshold
        if bp_change >= self.bp_threshold_buy:
            buy_signals += 1
        
        # ML regression prediction (if available)
        if self.model_trained and predicted_bp >= self.bp_threshold_buy:
            buy_signals += 1
        
        # ML classification prediction (if available)
        if self.model_trained and predicted_signal == "Buy":
            buy_signals += 1
        
        # Check for pyramid buying opportunity
        current_price = self._get_price_series()[-1] if self.price_history else 0
        is_pyramid_buy = False
        if self.model_trained and current_price > 0:
            is_pyramid_buy = self._should_pyramid_buy(current_price, predicted_bp)
        
        # Original buy logic OR pyramid buy
        total_possible = 4 if self.model_trained else 2
        majority_buy = buy_signals >= max(2, total_possible // 2)
        
        return majority_buy or is_pyramid_buy
    
    def _sell_conditions(self, ticker: str) -> bool:
        """Determine if sell conditions are met using ensemble of methods."""
        if len(self.price_history) < 10:  # Need some history
            return False
        
        # Get indicator signals (voting system)
        signals = self._get_indicator_signals()
        vote_result = self._voting_system(signals)
        
        # Check basis points change
        bp_change = self._calculate_basis_points_change()
        
        # Get ML predictions if available
        predicted_price = self._predict_next_price() if self.model_trained else 0
        predicted_signal = self._predict_signal() if self.model_trained else "Hold"
        
        # Calculate predicted price change in BP
        current_price = self._get_price_series()[-1] if self.price_history else 0
        predicted_bp = 0
        if current_price > 0 and predicted_price > 0:
            predicted_bp = ((predicted_price - current_price) / current_price) * 10000
        
        # Ensemble decision logic
        sell_signals = 0
        
        # Traditional indicator vote
        if vote_result == "Sell":
            sell_signals += 1
        
        # Current BP change below negative sell threshold
        if bp_change <= -self.bp_threshold_sell:
            sell_signals += 1
        
        # ML regression prediction (if available)
        if self.model_trained and predicted_bp <= -self.bp_threshold_sell:
            sell_signals += 1
        
        # ML classification prediction (if available)
        if self.model_trained and predicted_signal == "Sell":
            sell_signals += 1
        
        # Need majority agreement (at least 2 out of 4 possible signals)
        total_possible = 4 if self.model_trained else 2
        return sell_signals >= max(2, total_possible // 2)
    
class TradingBot:
    """Main trading bot that orchestrates the trading system."""
    
    def __init__(self, tickers: list[str], initial_capital: int):
        self.portfolio = StockPositions(tickers, initial_capital)
        self.tickers = tickers
        self.week_counter = 0
    
    def inject_market_data(self, ticker: str, timestamp, price: float, volume: float = 0) -> None:
        """Inject market data for simulation."""
        self.portfolio.inject_price_data(ticker, timestamp, price, volume)
    
    def get_portfolio_status(self) -> dict:
        """Get current portfolio status."""
        return {
            "current_capital": self.portfolio.current_cap,
            "portfolio_value": self.portfolio._portfolio_value_check(),
            "holdings": self.portfolio.current_own.copy(),
            "buyin_allocations": self.portfolio.buyin_portfolio.copy()
        }
    
    def get_trading_signals(self, ticker: str) -> dict:
        """Get current trading signals for a ticker."""
        return self.portfolio.get_current_signals(ticker)
    
    def execute_trading_cycle(self) -> dict:
        """Execute one trading cycle and return results."""
        results = {}
        
        # Execute trading logic
        self.portfolio.trade()
        
        # Get results for each ticker
        for ticker in self.tickers:
            results[ticker] = self.get_trading_signals(ticker)
        
        results["portfolio_status"] = self.get_portfolio_status()
        return results
    
    def finalize_week_and_retrain(self) -> dict:
        """Finalize current week's data and retrain all models."""
        self.week_counter += 1
        results = {}
        
        print(f"\n=== Finalizing Week {self.week_counter} and Retraining Models ===")
        
        for ticker in self.tickers:
            if (hasattr(self.portfolio, 'predictors') and 
                ticker in self.portfolio.predictors):
                
                predictor = self.portfolio.predictors[ticker]
                success = predictor.finalize_week_and_train()
                
                results[ticker] = {
                    'week': self.week_counter,
                    'retrained': success,
                    'weeks_in_history': len(predictor.weekly_training_data),
                    'model_ready': predictor.model_trained
                }
            else:
                results[ticker] = {
                    'week': self.week_counter,
                    'retrained': False,
                    'reason': 'no_predictor'
                }
        
        print(f"Week {self.week_counter} finalization complete\n")
        return results

# Faux portfolio integration function - will be replaced with real polygon.io integration
def create_production_portfolio(tickers: list[str], initial_capital: int, weeks_lookback: int = 8):
    """
    FAUX FUNCTION: This will be replaced with real polygon.io data collection.
    
    Production workflow:
    1. Collect 8+ weeks of historical data from polygon.io
    2. Train initial models on this data
    3. Deploy bot with trained models
    4. Weekly: collect new week's data, retrain, redeploy
    
    Args:
        tickers: List of stock symbols to trade
        initial_capital: Starting capital
        weeks_lookback: How many weeks of historical data to collect for initial training
    """
    print(f"=== PRODUCTION PORTFOLIO SETUP (FAUX) ===")
    print(f"Tickers: {tickers}")
    print(f"Capital: ${initial_capital:,}")
    print(f"Weeks lookback: {weeks_lookback}")
    
    # Create the trading bot
    bot = TradingBot(tickers, initial_capital)
    
    # FAUX: This would be replaced with polygon.io historical data collection
    print(f"\n[FAUX] Collecting {weeks_lookback} weeks of historical data from polygon.io...")
    
    # FAUX: Simulate collecting historical weekly data
    import random
    for week in range(weeks_lookback):
        print(f"[FAUX] Processing historical week {week + 1}/{weeks_lookback}")
        
        # FAUX: Generate some fake historical price data for each ticker
        for ticker in tickers:
            base_price = 100 + hash(ticker) % 200
            for day in range(5):  # 5 trading days per week
                for hour in range(6):  # 6 hours of trading per day
                    timestamp = week * 7 * 24 * 3600 + day * 24 * 3600 + hour * 3600
                    price = base_price * (1 + random.uniform(-0.02, 0.02))  # 2% daily volatility
                    volume = random.uniform(1000, 10000)  # Random volume
                    bot.inject_market_data(ticker, timestamp, price, volume)
        
        # At end of each historical week, finalize and train
        if week < weeks_lookback - 1:  # Don't finalize the last week yet
            bot.finalize_week_and_retrain()
    
    print(f"\n[FAUX] Historical data collection complete")
    print(f"[FAUX] Models trained on {weeks_lookback - 1} complete weeks")
    print(f"[FAUX] Bot ready for production trading")
    
    # Set up buy-in allocations (this would be configured elsewhere)
    allocation_per_ticker = initial_capital // len(tickers)
    allocations = {ticker: allocation_per_ticker for ticker in tickers}
    bot.portfolio.set_buyin_programmatically(allocations)
    
    return bot

def simulate_production_week(bot: TradingBot, week_number: int):
    """
    FAUX FUNCTION: Simulate one week of production trading.
    
    In production:
    1. Collect real-time data throughout the week
    2. Execute trading decisions
    3. At week end, retrain models and prepare for next week
    """
    print(f"\n=== SIMULATING PRODUCTION WEEK {week_number} ===")
    
    # FAUX: Simulate a week of trading
    for day in range(5):  # 5 trading days
        for ticker in bot.tickers:
            base_price = 100 + hash(ticker) % 200
            timestamp = week_number * 7 * 24 * 3600 + day * 24 * 3600
            price = base_price * (1 + random.uniform(-0.02, 0.02))
            volume = random.uniform(1000, 10000)
            
            # Inject real-time data
            bot.inject_market_data(ticker, timestamp, price, volume)
            
            # Execute trading if it's a trading hour
            if day % 2 == 0:  # Trade every other day for demo
                results = bot.execute_trading_cycle()
                
    # End of week: retrain models
    retrain_results = bot.finalize_week_and_retrain()
    
    # Show results
    portfolio_status = bot.get_portfolio_status()
    print(f"Week {week_number} complete:")
    print(f"  Portfolio value: ${portfolio_status['portfolio_value']:.2f}")
    print(f"  Cash: ${portfolio_status['current_capital']:.2f}")
    print(f"  Models retrained: {sum(1 for r in retrain_results.values() if r.get('retrained', False))} tickers")
    
    return portfolio_status, retrain_results

# Example usage for production simulation:
# 
# # Initial setup - collect historical data and train models
# bot = create_production_portfolio(["AAPL", "TSLA", "NVDA"], 50000, weeks_lookback=10)
# 
# # Simulate production trading weeks
# for week in range(1, 5):  # Simulate 4 weeks of production
#     portfolio_status, retrain_results = simulate_production_week(bot, week)
#     print(f"Week {week} P&L: ${portfolio_status['portfolio_value'] - 50000:.2f}")
#
# # In real production, this would be:
# # 1. polygon.io data collection running continuously
# # 2. Real trading API calls (Alpaca, etc.)
# # 3. Automated weekly retraining pipeline

# # 4. Risk management and monitoring systems
