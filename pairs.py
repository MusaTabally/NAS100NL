import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# Example asset prices: NASDAQ-100 futures vs. S&P 500 futures
nasdaq_futures = np.random.normal(12000, 50, 1000)  # Replace with actual data
sp500_futures = np.random.normal(3000, 20, 1000)    # Replace with actual data

# Check for cointegration
score, p_value, _ = coint(nasdaq_futures, sp500_futures)

print(f"Cointegration score: {score}, p-value: {p_value}")
if p_value < 0.05:  # Typically, p < 0.05 is the threshold for cointegration
    print("The assets are cointegrated and suitable for statistical arbitrage.")
else:
    print("The assets are not cointegrated.")
