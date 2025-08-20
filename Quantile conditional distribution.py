import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
tick_list = ['SPY', 'TLT', 'DBC', 'TIP', 'GLD', 'XLP', 'XLE', 'VNQ', 'XLU', 'XLV']
Tickers = yf.Tickers(tick_list).history(start="2005-06-17", end="2025-08-29", interval="1d") 

cp = Tickers["Close"].pct_change().dropna()
percentiles = [0.2, 0.4, 0.6, 0.8]

from collections import defaultdict


def get_quantiles(tickers, ret_df, percentiles):
    qs = defaultdict(list)
    for ticker in tickers:
        qs[ticker].extend(np.quantile(ret_df[ticker], percentiles))

    return qs

gc = get_quantiles(tick_list, cp, percentiles)

"""
Get the current two weeks's returns for each of the assets. Compute the quantiles for each of the assets. 
Find what quantile the returns belong in and condition based those quantiles to obtain the conditional 
distribution for the asset in question for the current market scenario.
"""

z = yf.Tickers(tick_list).history(start="2025-08-06", end="2025-08-29", interval="1d")["Close"].pct_change().mean()
print(z)

def fit_quantile(mean_rets, ticker_quantiles):
    """Matches the monthly returns to its quantile"""
    qu = defaultdict(list)
    for ticker, q_list in ticker_quantiles.items():
        mean_ret = mean_rets[ticker]
        print(f"mean ret for {ticker}: {mean_ret}")
        for i, q in enumerate(q_list):
            if mean_ret < q:
                if i==1:
                    qu[ticker].extend([-np.inf, q])
                else:
                    qu[ticker].extend([q_list[i-1], q_list[i]])
                break
            if i == len(q_list)-1:
                qu[ticker].extend([q_list[i], np.inf])

    return qu

current_quantiles = fit_quantile(z, gc)

mask_info = [("DBC", ">", 0.0116), ("XLU", ">", 0.0027), ("TLT", ">", 0.0024)]
lower_q = [(ticker, ">", val[0]) for ticker, val in current_quantiles.items()][:]
upper_q = [(ticker, "<", val[1]) for ticker, val in current_quantiles.items()][:2]
conds = lower_q + upper_q

print(conds)

from empiricaldist import Pmf

def cond_dist(cond_ticker, mask_info, ret_df):

    for ticker, action, action_val in mask_info:
        if ticker != cond_ticker:
            match action:
                case ">":
                    ret_df = ret_df[ret_df[ticker] > action_val]
                case "<":
                    ret_df = ret_df[ret_df[ticker] < action_val]

    dist = Pmf.from_seq([r for r in ret_df[cond_ticker].round(2)])
    dist.normalize()
    print(f"max prob: {dist.max_prob()}, mean: {dist.mean()}, std: {dist.std()}")

    return dist

plt.plot(cond_dist("SPY", conds, cp))
plt.show()
