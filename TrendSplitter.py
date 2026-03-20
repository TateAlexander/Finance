import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from collections import defaultdict

ticker = "AMZN"
start = "2026-01-01"

closes = yf.Ticker(ticker).history(start=start, end="2026-03-20", interval="1d")["Close"]

plt.plot(closes)
plt.show()

from scipy.stats import linregress, t

def find_break(closes, width=5):
    intervals_dict = defaultdict(list)
    interval_cnt = 1
    curr_slope_conf = None
    alpha = 0.05
    for i, _ in enumerate(closes[::width]):
        curr_interval = closes[i * width : (i + 1) * width]
        res = linregress([i for i in range(i * width, i * width + len(curr_interval))], curr_interval)

        slope = res[0]
        slope_std = res.stderr

        """For each slope param computed, break the interval if the current slope param is out of the prior slope param's alpha confidence interval"""

        if i == 0:
            intervals_dict[f"interval {interval_cnt}"].extend(curr_interval)
            t_crit = t.ppf(1-alpha/2, len(curr_interval) - 2)
            curr_slope_conf = (slope - t_crit * slope_std, slope + t_crit * slope_std)
        else:
            if slope < curr_slope_conf[0] or slope > curr_slope_conf[1]:
                t_crit = t.ppf(1-alpha/2, len(curr_interval) - 2)
                curr_slope_conf = (slope - t_crit * slope_std, slope + t_crit * slope_std)
                interval_cnt += 1
            
            intervals_dict[f"interval {interval_cnt}"].extend(curr_interval)

    return intervals_dict


for n, intervals in find_break(closes).items():
    plt.plot(intervals)
    plt.show()
