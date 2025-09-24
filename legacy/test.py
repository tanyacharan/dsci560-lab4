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

    def __init__(self, stock_names: list[str], init_cap: int) -> None:
        self.stock_names = stock_names
        # (stock name, max buy-in)
        self.buyin_portfolio = dict(zip(stock_names, [0] * len(stock_names)))
        # (stock name, current number of stocks owned) - we assume no previous ownership
        self.current_own = dict(zip(stock_names, [0] * len(stock_names)))
        self.init_cap = init_cap
        self.current_cap = init_cap
        self.portfolio_value = init_cap
        # This is probably not optimal but whatever - TODO: consider?
        self.platform = AlpacaConstraints

    def set_buyin(self, stock_names: list[str]) -> None:
        for name in stock_names:
            buyin = int(input(f"Input your buy-in capital for ticker {name}:"))
            self.buyin_portfolio[name] = buyin
        print("Buy-in capital setup success!")

    def print_buyin(self) -> None:
        print("Buy-in capital of current stocks:")
        for name, price in self.buyin_portfolio:
            print(f"Ticker {name} buy-in: ${price}")

    def _stock_price_check(self, stock_name: str) -> int:
        # TODO: Extract stock price here
        return 100
    
    def _portfolio_value_check(self) -> int:
        # TODO: Update
        return sum((0, 0))
    
    def _transaction_costs_buy(self, count: int) -> float:
        comms = self.platform.COMMISSION_PER_SHARE * count
        return comms

    def _action_buy(self, stock_name: str) -> None:
        stock_price = self._stock_price_check(stock_name)
        count = int(max(self.buyin_portfolio[stock_name], self.current_cap)/stock_price)
        if count == 0:
            print("Cannot proceed with purchase - not enough funds.")
        # Technically for all steps below you need to ensure that the transaction goes through
        # TODO: If we ever connect to API, remember to assert() and do whatever else is needed
        cost = round(count * stock_price, 4) + self._transaction_costs_buy(count)
        self.current_own[stock_name] += count
        self.current_cap -= cost
        # Warning system post-buy
        if self.current_cap <= 0.1 * self.init_cap:
            print("Warning: Low funds.")

    def _transaction_costs_sell(self, count: int) -> float:
        comms = self.platform.COMMISSION_PER_SHARE * count
        finra_taf = self.platform.FINRA_TAF_FEE * count
        sec_fee = self.platform.SEC_FEE * count
        return comms + finra_taf + sec_fee
        
    def _action_sell(self, stock_name: str) -> None:
        stock_price = self._stock_price_check(stock_name)
        # We assume we will sell all stocks.
        count = self.current_own[stock_name]
        # Technically for all steps below you need to ensure that the transaction goes through
        # TODO: If we ever connect to API, remember to assert() and do whatever else is needed
        earnings = round(count * stock_price, 4) - self._transaction_costs_sell(count)
        self.current_own[stock_name] = 0
        self.current_cap += earnings
        
    def _action_hold(self, stock_name: str) -> None:
        if self.current_own[stock_name] > 0:
            print(f"Holding {self.current_own[stock_name]} of stock {stock_name}.")

    def trade(self, stock_names: list[str]) -> None:
        """Trade stocks if sufficient requirements are made."""
        # TODO: Figure this out
        for name in stock_names:
            if TradingPredictors._buy_conditions(name):
                self._action_buy(name)
            elif TradingPredictors._sell_conditions(name):
                self._action_sell(name)
            else:
                self._action_hold(name)
        print(f"Current capital: {self.current_cap}")
        print(f"Current stock holdings:")
        for name in stock_names:
            print(self.current_own[name])
    
    # TODO: I guess we're not doing margin trading for now, but note that for later.

class TradingPredictors:
    def __init__(self, stock_name: str) -> None:
        pass
    
    def _calculate_macd(self) -> list[int]:
        return [0]
    
    def _calculate_rsi(self) -> list[int]:
        return [0]
    
    def _calculate_whatever(self) -> int:
        return 0
    
    def _buy_conditions(self, stock_name: str) -> bool:
        if self._calculate_whatever() > 70 and self._calculate_rsi()[0] > 70:
            return True
        return False
    
    def _sell_conditions(self, stock_name: str) -> bool:
        if self._calculate_whatever() > 70 and self._calculate_rsi()[0] > 70:
            return True
        return False