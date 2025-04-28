#!/usr/bin/env python
# coding: utf-8

"""
Assignment 3 Solution Script
This script implements the solutions for Assignment 3 of the QPMwP course.
"""

# Standard library imports
import os
import sys
import types

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root directory to Python path
project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Local modules imports
from helper_functions import load_data_msci
from estimation.covariance import Covariance
from estimation.expected_return import ExpectedReturn
from optimization.optimization import MeanVariance
from backtesting.backtest_item_builder_classes import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
)
from backtesting.backtest_item_builder_functions import (
    bibfn_selection_data_random,
    bibfn_return_series,
    bibfn_budget_constraint,
    bibfn_box_constraints,
)
from backtesting.portfolio import floating_weights
from backtesting.backtest_service import BacktestService
from backtesting.backtest import Backtest

print("Assignment 3 Solution")
print("====================")

# Load data
print("\n1. Loading data...")
N = 24
data = load_data_msci(path='data/', n=N)

# Define rebalancing dates
print("\n2. Setting up backtest configuration...")
n_days = 21 * 3
start_date = '2010-01-01'
dates = data['return_series'].index
rebdates = dates[dates > start_date][::n_days].strftime('%Y-%m-%d').tolist()

# Define the selection item builders.
selection_item_builders = {
    'data': SelectionItemBuilder(
        bibfn=bibfn_selection_data_random,
        k=10,
        seed=42,
    ),
}

# Define the optimization item builders.
optimization_item_builders = {
    'return_series': OptimizationItemBuilder(
        bibfn=bibfn_return_series,
        width=365 * 3,
    ),
    'budget_constraint': OptimizationItemBuilder(
        bibfn=bibfn_budget_constraint,
        budget=1,
    ),
    'box_constraints': OptimizationItemBuilder(
        bibfn=bibfn_box_constraints,
        upper=0.5,
    ),
}

# Initialize the backtest service
bs = BacktestService(
    data=data,
    selection_item_builders=selection_item_builders,
    optimization_item_builders=optimization_item_builders,
    optimization=MeanVariance(
        covariance=Covariance(method='pearson'),
        expected_return=ExpectedReturn(method='geometric'),
        risk_aversion=1,
        solver_name='cvxopt',
    ),
    rebdates=rebdates,
)

# Instantiate the backtest object and run the backtest
print("\n3. Running backtest...")
bt_mv = Backtest()
bt_mv.run(bs=bs)

# Define turnover function (solution to Question 1)
print("\n4. Implementing turnover function (Question 1)...")
def turnover(self, return_series: pd.DataFrame, rescale: bool=True):
    """
    Calculate the turnover between consecutive portfolios.
    
    Args:
        return_series: DataFrame with returns of all assets
        rescale: Boolean indicating whether to rescale the weights
        
    Returns:
        Series with turnover values for each rebalancing date
    """
    dates = self.get_rebalancing_dates()
    to = {}
    to[dates[0]] = float(1)
    for rebalancing_date in dates[1:]:

        previous_portfolio = self.get_previous_portfolio(rebalancing_date)
        current_portfolio = self.get_portfolio(rebalancing_date)
        
        if current_portfolio.rebalancing_date is None or previous_portfolio.rebalancing_date is None:
            raise ValueError('Portfolios must have a rebalancing date')

        if current_portfolio.rebalancing_date < previous_portfolio.rebalancing_date:
            raise ValueError('The previous portfolio must be older than the current portfolio')

        # Get the union of the ids of the weights in both portfolios (previous and current)
        all_ids = set(previous_portfolio.weights.index) | set(current_portfolio.weights.index)

        # Extend the weights of the previous portfolio to the union of ids in both portfolios by adding zeros
        prev_weights_extended = pd.Series(0, index=all_ids)
        prev_weights_extended.loc[previous_portfolio.weights.index] = previous_portfolio.weights

        # Float the weights of the previous portfolio according to the price drifts in the market 
        # until the current rebalancing date
        floated_weights = floating_weights(
            X=return_series,
            w=prev_weights_extended,
            start_date=previous_portfolio.rebalancing_date,
            end_date=current_portfolio.rebalancing_date,
            rescale=rescale
        ).iloc[-1]

        # Extract the weights of the current portfolio
        current_weights_extended = pd.Series(0, index=all_ids)
        current_weights_extended.loc[current_portfolio.weights.index] = current_portfolio.weights

        # Calculate the turnover as the sum of absolute differences between floated weights and current weights
        to[rebalancing_date] = np.sum(np.abs(floated_weights - current_weights_extended))

    return pd.Series(to)

# Define simulate function (solution to Question 2)
print("\n5. Implementing simulation function (Question 2)...")
def simulate(self,
            return_series: pd.DataFrame,
            fc: float = 0,
            vc: float = 0,
            n_days_per_year: int = 252) -> pd.Series:
    """
    Simulate portfolio returns with transaction costs.
    
    Args:
        return_series: DataFrame with returns of all assets
        fc: Fixed cost as annual percentage
        vc: Variable cost as percentage of turnover
        n_days_per_year: Number of trading days per year
        
    Returns:
        Series with portfolio returns
    """
    rebdates = self.get_rebalancing_dates()
    ret_list = []
    for rebdate in rebdates:
        next_rebdate = (
            rebdates[rebdates.index(rebdate) + 1]
            if rebdate < rebdates[-1]
            else return_series.index[-1]
        )

        portfolio = self.get_portfolio(rebdate)
        w_float = portfolio.float_weights(
            return_series=return_series,
            end_date=next_rebdate,
            rescale=False # Notice that rescale is hardcoded to False.
        )
        level = w_float.sum(axis=1)
        ret_tmp = level.pct_change(1)
        ret_list.append(ret_tmp)

    portf_ret = pd.concat(ret_list).dropna()

    if vc != 0:
        # Calculate turnover
        to = self.turnover(return_series=return_series,
                           rescale=False)
        
        # Calculate variable cost (vc) as a fraction of turnover and
        # subtract the variable cost from the returns at each rebalancing date
        for rebdate in rebdates:
            if rebdate in to.index and rebdate in portf_ret.index:
                # Apply variable cost as percentage of turnover on rebalancing dates
                portf_ret.loc[rebdate] -= to[rebdate] * vc

    if fc != 0:
        # Calculate number of days between returns
        days_between = pd.Series(index=portf_ret.index)
        for i in range(len(days_between)-1):
            days_between.iloc[i+1] = (portf_ret.index[i+1] - portf_ret.index[i]).days
        days_between.iloc[0] = 1  # Assume 1 day for the first entry
        
        # Calculate daily fixed cost based on the annual fixed cost (fc),
        # the number of days between two rebalancings and the number of days per year
        daily_fc = fc / n_days_per_year
        
        # Subtract the daily fixed cost from the daily returns
        portf_ret = portf_ret - daily_fc

    return portf_ret

# Overwrite the turnover method of the strategy object
bt_mv.strategy.turnover = types.MethodType(turnover, bt_mv.strategy)

# Overwrite the simulate method of the strategy object
bt_mv.strategy.simulate = types.MethodType(simulate, bt_mv.strategy)

# Calculate and plot the turnover
print("\n6. Calculating and plotting turnover...")
to = bt_mv.strategy.turnover(
    return_series=data['return_series'],
    rescale=True
)
plt.figure(figsize=(10, 5))
to.plot(title='Turnover')
plt.savefig('turnover.png')
plt.close()

# Simulate with different cost assumptions
print("\n7. Running simulations with different cost assumptions...")
return_series = bs.data['return_series']

sim_mv_gross = bt_mv.strategy.simulate(return_series=return_series, fc=0, vc=0)
sim_mv_net_of_fc = bt_mv.strategy.simulate(return_series=return_series, fc=0.01, vc=0)
sim_mv_net_of_vc = bt_mv.strategy.simulate(return_series=return_series, fc=0, vc=0.002)
sim_mv_net = bt_mv.strategy.simulate(return_series=return_series, fc=0.01, vc=0.002)

# Plot the cumulative returns
sim = pd.concat({
    'mv_gross': sim_mv_gross,
    'mv_net_of_fc': sim_mv_net_of_fc,
    'mv_net_of_vc': sim_mv_net_of_vc,
    'mv_net': sim_mv_net,
}, axis=1).dropna()

plt.figure(figsize=(10, 6))
np.log((1 + sim)).cumsum().plot()
plt.title('Cumulative Log Returns')
plt.legend(['Gross', 'Net of FC', 'Net of VC', 'Net'])
plt.savefig('cumulative_returns.png')
plt.close()

# Define a function to calculate performance metrics (solution to Question 3)
print("\n8. Calculating performance metrics (Question 3)...")
def calculate_performance_metrics(returns, n_days_per_year=252):
    """
    Calculate various performance metrics for a return series.
    
    Args:
        returns: Series of returns
        n_days_per_year: Number of trading days per year
    
    Returns:
        Series with performance metrics
    """
    # Ensure we have a clean Series
    returns = returns.dropna()
    
    # Calculate cumulative return
    cum_return = (1 + returns).prod() - 1
    
    # Calculate annualized return
    n_days = len(returns)
    ann_return = (1 + cum_return) ** (n_days_per_year / n_days) - 1
    
    # Calculate annualized volatility
    ann_vol = returns.std() * np.sqrt(n_days_per_year)
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    
    # Calculate maximum drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    return pd.Series({
        'Cumulative Return': f"{cum_return:.4f}",
        'Annualized Return': f"{ann_return:.4f}",
        'Annualized Volatility': f"{ann_vol:.4f}",
        'Sharpe Ratio': f"{sharpe:.4f}",
        'Maximum Drawdown': f"{max_drawdown:.4f}"
    })

# Create a dictionary to store performance metrics for each simulation
performance = {}

# Calculate performance metrics for each simulation
performance['Mean-Variance (Gross)'] = calculate_performance_metrics(sim_mv_gross)
performance['Mean-Variance (Net of FC)'] = calculate_performance_metrics(sim_mv_net_of_fc)
performance['Mean-Variance (Net of VC)'] = calculate_performance_metrics(sim_mv_net_of_vc)
performance['Mean-Variance (Net)'] = calculate_performance_metrics(sim_mv_net)

# Create a DataFrame with the performance metrics
performance_df = pd.DataFrame(performance).T

# Print the performance metrics
print("\nPerformance Metrics:")
print(performance_df)

# Save the performance metrics to a CSV file
performance_df.to_csv('performance_metrics.csv')

print("\nTurnover statistics:")
print(to.describe())

print("\nAssignment 3 solution completed! Results saved to turnover.png, cumulative_returns.png, and performance_metrics.csv")