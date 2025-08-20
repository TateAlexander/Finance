import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
tick_list = ['SPY', 'TLT', 'DBC', 'TIP', 'GLD', 'XLP', 'XLE', 'VNQ', 'XLU', 'XLV']
Tickers = yf.Tickers(tick_list).history(start="2005-06-17", end="2025-08-29", interval="1d") 

cp = Tickers["Close"].pct_change().dropna()
mask_info = [("DBC", ">", 0), ("XLU", "<", 0)]


from empiricaldist import Pmf

def cond_dist(cond_ticker, mask_info, ret_df):

    for ticker, action, action_val in mask_info:
        match action:
            case ">":
                ret_df = ret_df[ret_df[ticker] > action_val]
            case "<":
                ret_df = ret_df[ret_df[ticker] < action_val]

    dist = Pmf.from_seq([r for r in ret_df[cond_ticker].round(2)])
    dist.normalize()
    print(f"max prob: {dist.max_prob()}, mean: {dist.mean()}, std: {dist.std()}")

    return dist

plt.plot(cond_dist("SPY", mask_info, cp))
plt.show()
