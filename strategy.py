# hola madrid
# Strategy Type: custom
# Description: hola madrid
# Created: 2025-06-14T17:31:07.384Z

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the HolaMadridStrategy class
class HolaMadridStrategy:
    def __init__(self, data, short_window=40, long_window=100):
        # Initialize the strategy with data and parameters
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signals = pd.DataFrame(index=data.index)
        self.signals['signal'] = 0.0

    def generate_signals(self):
        # Generate signals based on moving average crossover
        self.signals['short_mavg'] = self.data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        self.signals['long_mavg'] = self.data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        self.signals['signal'][self.short_window:] = np.where(self.signals['short_mavg'][self.short_window:] 
                                                              > self.signals['long_mavg'][self.short_window:], 1.0, 0.0)
        self.signals['positions'] = self.signals['signal'].diff()
        return self.signals

    def backtest_strategy(self, initial_capital=10000.0):
        # Backtest the strategy
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions['Asset'] = 100 * self.signals['signal']
        portfolio = positions.multiply(self.data['Close'], axis=0)
        pos_diff = positions.diff()
        portfolio['holdings'] = (positions.multiply(self.data['Close'], axis=0)).sum(axis=1)
        portfolio['cash'] = initial_capital - (pos_diff.multiply(self.data['Close'], axis=0)).sum(axis=1).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio

    def calculate_performance_metrics(self, portfolio):
        # Calculate performance metrics
        returns = portfolio['returns']
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
        drawdown = (portfolio['total'].cummax() - portfolio['total']) / portfolio['total'].cummax()
        max_drawdown = drawdown.max()
        return sharpe_ratio, max_drawdown

# Generate sample data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500, freq='D')
prices = np.random.lognormal(mean=0.001, sigma=0.02, size=len(dates))
data = pd.DataFrame(data={'Close': prices}, index=dates)

# Main execution block
if __name__ == "__main__":
    # Initialize the strategy
    strategy = HolaMadridStrategy(data)
    # Generate signals
    signals = strategy.generate_signals()
    # Backtest the strategy
    portfolio = strategy.backtest_strategy()
    # Calculate performance metrics
    sharpe_ratio, max_drawdown = strategy.calculate_performance_metrics(portfolio)
    # Print performance metrics
    print("Sharpe Ratio:", sharpe_ratio)
    print("Max Drawdown:", max_drawdown)
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio['total'], label='Portfolio value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

# Strategy Analysis and Performance
# Add your backtesting results and analysis here

# Risk Management
# Document your risk parameters and constraints

# Performance Metrics
# Track your strategy's key performance indicators:
# - Sharpe Ratio
# - Maximum Drawdown
# - Win Rate
# - Average Return
