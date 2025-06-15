"""Capture a stocks up trend by connecting the lows that are lower than its two neighbors"""
import numpy as np
import matplotlib.pyplot as plt
import yfinance as fy

security = fy.ticker.Ticker("DAC").history(interval='1d', start='2025-04-01', end="2025-06-13")

def neighboringLows(stock_df):
    nls = []
    for i, low in enumerate(stock_df["Low"]):
        if i > 0 and i < len(stock_df["Low"])-1 :
            if low < stock_df["Low"][i-1] and low < stock_df["Low"][i+1]:
                nls.append((i, low))

    trends = []
    trend = []
    for nls_i, (low_i, low) in enumerate(nls):
        if nls_i < len(nls)-1:
            if low < nls[nls_i+1][1]:
                if len(trend) == 0:
                    trend.append(nls[nls_i])
                trend.append(nls[nls_i+1])
            else:
                if len(trend) > 0:
                    trends.append(trend)
                    trend = []

    return trends

print(neighboringLows(security))

plt.plot(security["Low"])
plt.show()
