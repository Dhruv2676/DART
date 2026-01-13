import numpy as np
import pandas as pd

class TradingStrategy:
    """
    Implements a basic backtesting engine for stock trading strategies.
    Simulates portfolio performance based on model predictions and calculates financial metrics.
    """
    def __init__(self, initial_balance=1_000_000.0, transaction_cost=0.001, risk_free_rate=0.0, annualization_factor=252):
        """
        Initializes the trading strategy parameters.

        Parameters:
        - initial_balance: float, starting capital.
        - transaction_cost: float, percentage of turnover per trade (e.g., 0.001 = 0.1%).
        - risk_free_rate: float, annual risk-free rate for Sharpe ratio calculation.
        - annualization_factor: int, trading days in a year (default 252).
        """
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        
    def run_backtest(self, predictions, actual_returns, num_companies, dates=None, down_class_index=0, up_class_index=2):
        """
        Executes the backtest simulation.

        Parameters:
        - predictions: np.array, flattened array of model predictions (probabilities).
        - actual_returns: np.array, flattened array of actual log returns.
        - num_companies: int, number of assets in the universe.
        - dates: list, corresponding dates for each timestep.
        - down_class_index: int, index for 'Sell'/'Down' class in prediction vector.
        - up_class_index: int, index for 'Buy'/'Up' class in prediction vector.

        Returns:
        - pd.Series: Daily log returns of the portfolio.
        - list: History of total portfolio value over time.
        - list: List of dictionaries containing date and return for each step.
        """
        num_samples = len(predictions)
        if num_samples == 0:
            return pd.Series([], dtype=float), [self.initial_balance]
        num_timesteps = num_samples // num_companies
        
        try:
            # Reshape flat arrays into (Time, Company, Features)
            reshaped_predictions = predictions.reshape(num_timesteps, num_companies, -1)
            reshaped_returns = actual_returns.reshape(num_timesteps, num_companies)
        except ValueError as e:
            print(f"Error reshaping arrays. Total samples: {num_samples}, Num companies: {num_companies}.")
            print(f"Make sure (num_samples % num_companies) == 0. Error: {e}")
            return pd.Series([], dtype=float), [self.initial_balance]
        
        # Simulate price paths starting from 100.0
        prices = np.ones((num_timesteps + 1, num_companies)) * 100.0
        for t in range(num_timesteps):
            prices[t+1] = prices[t] * np.exp(reshaped_returns[t])
            
        balance = self.initial_balance
        shares_held = np.zeros(num_companies)
        portfolio_value_history = [self.initial_balance]
        daily_returns_log = []
        daily_returns_with_dates = []
        
        for t in range(num_timesteps):
            current_day_prices = prices[t]
            current_total_value = portfolio_value_history[-1]
            
            # Simple Strategy: Weight = P(Up) - P(Down)
            current_probs = reshaped_predictions[t]
            scores = current_probs[:, up_class_index] - current_probs[:, down_class_index]
            scores[scores < 0] = 0.0 # Long-only strategy
            total_score = np.sum(scores)
            
            if total_score > 1e-6:
                target_weights = scores / total_score
            else:
                target_weights = np.zeros(num_companies)
            
            # Rebalance portfolio
            target_capital_allocation = current_total_value * target_weights
            valid_prices = current_day_prices.copy()
            valid_prices[valid_prices == 0] = 1e-6 # Avoid division by zero
            target_shares = target_capital_allocation / valid_prices
            
            trades = target_shares - shares_held
            transaction_value = np.sum(np.abs(trades) * current_day_prices)
            transaction_cost = transaction_value * self.transaction_cost
            
            # Update balance and holdings
            balance = balance - (np.sum(trades * current_day_prices)) - transaction_cost
            shares_held = target_shares
            end_of_day_prices = prices[t+1]
            
            # Calculate new portfolio value
            new_value = balance + np.sum(shares_held * end_of_day_prices)
            
            portfolio_value_history.append(new_value)
            daily_return = (new_value - current_total_value) / current_total_value
            daily_returns_log.append(daily_return)
            
            if dates is not None and t < len(dates):
                daily_returns_with_dates.append({
                    'date': dates[t],
                    'return': float(daily_return)
                })
            else:
                daily_returns_with_dates.append({
                    'date': f'timestep_{t}',
                    'return': float(daily_return)
                })
        
        return pd.Series(daily_returns_log, dtype=float), portfolio_value_history, daily_returns_with_dates
    
    def calculate_metrics(self, strategy_returns_series, portfolio_value_history, benchmark_returns_series=None):
        """
        Computes standard financial performance metrics.

        Parameters:
        - strategy_returns_series: pd.Series, daily returns of the strategy.
        - portfolio_value_history: list, sequence of total portfolio values.
        - benchmark_returns_series: pd.Series (optional), daily returns of a benchmark for Beta/Alpha.

        Returns:
        - dict: containing Sharpe, Sortino, Max Drawdown, Alpha, Beta, etc.
        """
        returns = strategy_returns_series.astype(float)
        if returns.empty or len(returns) < 2 or len(portfolio_value_history) < 2:
            return {'cumulative_returns': 0.0, 'final_value': self.initial_balance, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 
                    'sortino_ratio': 0.0, 'alpha': 0.0, 'beta': 0.0, 'mean_return': 0.0, 'volatility': 0.0}
        
        final_value = portfolio_value_history[-1]
        cumulative_returns = (final_value - self.initial_balance) / self.initial_balance
        
        mean_return_annual = returns.mean() * self.annualization_factor
        
        volatility_annual = returns.std() * np.sqrt(self.annualization_factor)
        volatility_annual = 0.0 if np.isnan(volatility_annual) else volatility_annual
        
        daily_risk_free = self.risk_free_rate / self.annualization_factor
        excess_returns = returns - daily_risk_free
        
        # Sharpe Ratio
        if excess_returns.std() == 0 or np.isnan(excess_returns.std()):
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(self.annualization_factor)
            sharpe_ratio = 0.0 if np.isnan(sharpe_ratio) else sharpe_ratio
        
        # Sortino Ratio (Downside deviation only)
        downside_returns = excess_returns[excess_returns < 0]
        if downside_returns.empty:
            sortino_ratio = 0.0
        else:
            downside_std = downside_returns.std()
            if downside_std == 0 or np.isnan(downside_std):
                sortino_ratio = 0.0
            else:
                sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(self.annualization_factor) 
                sortino_ratio = 0.0 if np.isnan(sortino_ratio) else sortino_ratio
        
        # Max Drawdown
        pv_series = pd.Series(portfolio_value_history)
        peak = pv_series.expanding(min_periods=1).max()
        drawdown = (pv_series - peak) / peak
        max_drawdown = drawdown.min()
        max_drawdown = 0.0 if np.isnan(max_drawdown) else max_drawdown
        
        # Alpha and Beta (if benchmark provided)
        alpha = 0.0
        beta = 0.0
        if benchmark_returns_series is not None:
            df = pd.DataFrame({'strategy': returns, 'benchmark': benchmark_returns_series}).dropna()
            if not df.empty and len(df) > 1:
                variance_benchmark = df['benchmark'].var()
                if variance_benchmark > 1e-6:
                    cov_matrix = df.cov()
                    beta = cov_matrix.loc['strategy', 'benchmark'] / variance_benchmark
                    alpha = mean_return_annual - (self.risk_free_rate + beta * (df['benchmark'].mean() * self.annualization_factor - self.risk_free_rate))
                
                alpha = 0.0 if np.isnan(alpha) else alpha
                beta = 0.0 if np.isnan(beta) else beta
            
        return {'cumulative_returns': float(cumulative_returns),
                'final_value': float(final_value),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'sortino_ratio': float(sortino_ratio),
                'alpha': float(alpha),
                'beta': float(beta),
                'mean_return': float(mean_return_annual),
                'volatility': float(volatility_annual)}
    
    def average_run_metrics(self, list_of_metric_dicts):
        """
        Calculates the average of each metric across multiple simulation runs.
        
        Parameters:
        - list_of_metric_dicts: list of dicts, where each dict contains metrics from one run.
        
        Returns:
        - dict: averaged metrics.
        """
        if not list_of_metric_dicts:
            return {}
        
        avg_metrics = {}
        all_keys = list_of_metric_dicts[0].keys()
        for key in all_keys:
            values = [d[key] for d in list_of_metric_dicts if key in d]
            if values:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics