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

#-----------------------------
#    Updated version below:

"""Capture a stocks up trend by connecting the lows that are lower than its two neighbors"""
import numpy as np
import matplotlib.pyplot as plt
import yfinance as fy

security = fy.ticker.Ticker("TQQQ").history(interval='1d', start='2025-04-01', end="2025-06-13")

def neighboringLows(stock_df):
    nls = []
    for i, low in enumerate(stock_df["Low"]):
        if i > 0 and i < len(stock_df["Low"])-1 :
            if low < stock_df["Low"][i-1] and low < stock_df["Low"][i+1]:
                nls.append((i, low))

    trends = []
    trend = []
    closeUpTrends = []
    for nls_i, (low_i, low) in enumerate(nls):
        if nls_i < len(nls)-1:
            if low < nls[nls_i+1][1]:
                if len(trend) == 0:
                    trend.append(nls[nls_i])

                trend.append(nls[nls_i+1])
            else:
                if len(trend) > 0:
                    trends.append(trend)
                    closeUpTrends.append([x[0] for x in trend])
                    trend = []

    return closeUpTrends

def makeTrendLine(cuts, stock_df):
    t = []
    for trend in cuts:

        uptrend = stock_df["Close"][trend[0]: trend[-1]]
        uptrendStd = uptrend.std()
        x = [num for num in np.arange(trend[0], trend[-1], 1)]
        reg = np.polyfit(x, uptrend, deg=1)
        trend = np.polyval(reg, x)

        plt.plot(x, trend, 'r')
        plt.plot(x, trend + uptrendStd, 'g--')
        plt.plot(x, trend - uptrendStd, 'g--')
        plt.plot(x, uptrend)
        plt.show()

nl = neighboringLows(security)

print(nl)
print(makeTrendLine(nl, security))
#-----------------------------------------------------------------------------------------------------------
#     ANOTHA IMPROVEMENT BELOW!!!!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import geom

trendLengths = np.linspace(1, 11, 11)
g = geom(1/3)

def sim(geomModel, gain, loss, rounds, take):
    """ Simulating if taking a percentage of capital out of the investment each day would make an otherwise unprofitable investment profitable"""
    intervals = geomModel.rvs(rounds)
    avgProfit = 0

    for intervalLength in intervals:
        keep = 0
        inMarket = 1

        for _ in range(1, intervalLength+1):
            inMarket *= gain
            #   Take a percentage of the money out 
            a = take * inMarket
            keep += a
            inMarket -= a
        
        keep += inMarket*loss
        avgProfit += keep

    return avgProfit/len(intervals)

conserv_perc = np.round(np.linspace(0, 1, 100), 2)

def multiSim(conserv_perc):
    optimal = []
    for perc in conserv_perc:
        optimal.append((perc, sim(g, 1.05, 0.9, 100_000, perc)))

    return optimal

# optimal = sorted(multiSim(conserv_perc), key=lambda x:x[1])
# print(optimal)
# plt.plot(optimal)
# plt.show()
#   ----------------------------------------------------------------
#   Linear Regression

# import yfinance as yf

# security = yf.ticker.Ticker("XLU").history(interval='1d', start='2025-04-01', end='2025-06-13')
# print(security["Close"])
# x = [num for num in range(len(security["Close"]))]
# reg = np.polyfit(x, security['Close'], deg=1)
# trend = np.polyval(reg, x)
# std = security['Close'].std()
# plt.scatter(x, security['Close']) 
# plt.plot(x, trend, 'r')
# plt.plot(x, trend + std, 'g--')
# plt.plot(x, trend - std, 'g--')
# plt.show()
#---------------------------------
"""Capture a stocks up trend by connecting the lows that are lower than its two neighbors"""
import numpy as np
import matplotlib.pyplot as plt
import yfinance as fy

security = fy.ticker.Ticker("AMZN").history(interval='1d', start='2025-04-01', end="2025-06-13")

def neighboringLows(stock_df):
    nls = []
    for i, low in enumerate(stock_df["Low"]):
        if i > 0 and i < len(stock_df["Low"])-1 :
            if low < stock_df["Low"][i-1] and low < stock_df["Low"][i+1]:
                nls.append((i, low))

    trends = []
    trend = []
    closeUpTrends = []
    for nls_i, (low_i, low) in enumerate(nls):
        if nls_i < len(nls)-1:
            if low < nls[nls_i+1][1]:
                if len(trend) == 0:
                    trend.append(nls[nls_i])

                trend.append(nls[nls_i+1])
            else:
                if len(trend) > 0:
                    trends.append(trend)
                    closeUpTrends.append([x[0] for x in trend])
                    trend = []

    return closeUpTrends

def makeTrendLine(cuts, stock_df):
    t = []
    for trend in cuts:

        uptrend = stock_df["Close"][trend[0]: trend[-1]]
        uptrendStd = uptrend.std()
        x = [num for num in np.arange(trend[0], trend[-1], 1)]
        reg = np.polyfit(x, uptrend, deg=1)
        trend = np.polyval(reg, x)

        plt.plot(x, trend, 'r')
        plt.plot(x, trend + uptrendStd, 'g--')
        plt.plot(x, trend - uptrendStd, 'g--')

    plt.plot([x for x in range(len(stock_df["Close"]))], stock_df["Close"])
    plt.show()


nl = neighboringLows(security)

print(nl)
print(makeTrendLine(nl, security))


