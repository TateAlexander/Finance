import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yfinance as yf
from empiricaldist import Pmf
from collections import defaultdict
import re

class Price_Data:
    def __init__(self, stock, start_date="2025-05-01", end_date="2025-06-29", price_type="Close"):  
        self.stock = stock
        self.stock_df = yf.Ticker(stock).history(interval="1d", start=start_date, end=end_date)
        self.price_type = price_type
        #   Mark a share price increase with 1 & decrease with 0
        self.stock_df["Change"] = self.stock_df[self.price_type].transform(lambda x: x.diff().apply(lambda d: 1 if d > 0 else 0))

    
    def ema(self, days=14):
        self.stock_df["Ema"] = self.stock_df[self.price_type].transform(lambda x: x.ewm(span=days, adjust=True).mean())

    def calRSI(self, daysInterval=2):
        """Calculates the RSI for the days interval provided"""

        log_returns = np.log(self.stock_df[self.price_type]/self.stock_df[self.price_type].shift()).dropna()
            
        positive = log_returns.copy()
        negative = log_returns.copy()

        positive[positive < 0] = 0
        negative[negative > 0] = 0

        days = daysInterval

        averageGain = positive.rolling(window=days).mean()
        averageLoss = abs(negative.rolling(window=days).mean())

        # print(averageGain, averageLoss)

        relativeStrength = averageGain / averageLoss
        relativeStrength[relativeStrength.isna()] = 0

        RSI = 100 - (100 /(1 + relativeStrength))
        
        self.stock_df['RSI'] = np.round(RSI, 2)

    def cal_macd(self, short_days=1, long_days=4):
        short_sig = self.stock_df[self.price_type].transform(lambda x: x.ewm(span=short_days).mean())
        self.stock_df["Short"] = short_sig

        long_sig = self.stock_df[self.price_type].transform(lambda x: x.ewm(span=long_days).mean())
        self.stock_df["Long"] = long_sig

        self.stock_df["macd signal"] = short_sig - long_sig

        return (short_sig, long_sig)
    
    def macd_intervals(self):
        changes = [0]
        for i, signal in enumerate(self.stock_df["macd signal"]):
            if i >= 1:
                if (signal < 0 and self.stock_df["macd signal"][i-1] > 0) or (signal > 0 and self.stock_df["macd signal"][i-1] < 0):
                    if i != 0 and i != len(self.stock_df)-1:
                        changes.append(i)
        
        changes.append(len(self.stock_df)-1)
        
        subintervals = tuple([self.stock_df.iloc[changes[i]:changes[i+1]] for i in range(len(changes)-1)])
        
        return subintervals



# META = Price_Data("WOOF")
# DF = META.stock_df
# short, long = META.cal_macd()
# plt.plot(short)
# plt.plot(long)
# plt.show()

# for interval in META.macd_intervals():
#     print(interval.index, interval['macd signal'], interval['Change'])


#   I dont want frequent buy and sell signals. Longest interval with the most net up days is optimal. net negative intervals need to be penalized, but more so long net negative intervals.
signals = [(1,2), (1,3), (1,4), (1,6), (2,3), (2,4), (2,5), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (4, 7), (5,8), (7, 8), (7, 9), (8, 9), (8, 10), (9, 10), (9, 11), (10,20), (12, 26), (30, 40), (35,45), (35, 50), (40, 55)]
#Find the MACD SIGNAL combo with the highest avg sub interval length. Model the number of up days within that interval with a Geometric distribution

def b(signals):
    results = []
    for pair in signals:
        signal_total = 0
        META.cal_macd(pair[0], pair[1])
        for interval in META.macd_intervals():

            buySell = "buy below" if interval["macd signal"][len(interval)-1] > 0 else "sell below"
            if buySell == 'buy below':
                interval_total = 0
                changeToStr = [str(change) for change in interval["Change"]]
                combinedChange = ''.join(changeToStr)
                upSeq = [ele for ele in combinedChange.split('0') if ele != '']
                downSeq = [ele for ele in combinedChange.split('1') if ele != '']

                
                for seq in upSeq:
                    interval_total += 2**(len(seq)-1)

                for seq in downSeq:
                    interval_total -= 2**(len(seq)-1)

                signal_total += interval_total

                # print(buySell)
                # print(interval["macd signal"], interval["Change"], upSeq, downSeq, f"interval tot: {interval_total}")

        results.append((pair, signal_total))

    return results

# print(sorted(b(signals), key=lambda x:x[1]))

def buySell(tickers, start="2025-05-10", end="2025-06-20"):
    for ticker in tickers:
        t = Price_Data(ticker, start, end)
        """
        if prev day short > long and current day short < long then sell. Vice versa is buy.
        """
        short, long = t.cal_macd()

        if (short[-2] - long[-2]) < 0 and (short[-1] - long[-1]) > 0:
            print(f"Buy {ticker} on {t.stock_df.index[-2]}-{t.stock_df.index[-1]}")
        elif (short[-2] - long[-2]) > 0 and (short[-1] - long[-1]) < 0:
            print(f"Sell {ticker} on {t.stock_df.index[-2]}-{t.stock_df.index[-1]}")

        plt.plot(short)
        plt.plot(long)
        plt.show()



# optimal_macd(signals, dia)
t = ["ERX", "ERY", "TQQQ", "SQQQ", 'VOX', 'XLU', 'VNQ', 'GDX', 'XLI', 'XLV', 'XLF', 'XLE', 'XLP', 'XLY']
buySell(t)

