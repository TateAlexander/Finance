import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yfinance as yf
from empiricaldist import Pmf
from collections import defaultdict

class Price_Data:
    def __init__(self, stock, start_date="2023-01-15", end_date="2025-08-29", price_type="Close"):  
        self.stock = stock
        self.stock_df = yf.Ticker(stock).history(interval="1d", start=start_date, end=end_date)
        self.price_type = price_type
        #   Mark a share price increase with 1 & decrease with 0
        self.stock_df["Change"] = self.stock_df[self.price_type].transform(lambda x: x.diff().apply(lambda d: 1 if d > 0 else 0))
        self.stock_df["Returns"] = np.log(self.stock_df["Close"]/self.stock_df["Close"].shift())
    
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

    def cal_macd(self, short_days=15, long_days=30):
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


#!!!!!!!!!!!!!!!!!!!!!!!!!!
t = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "META", "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "XOM", "CVX", "HD", "DIS",
"BAC", "PFE", "KO", "PEP", "NFLX", "CSCO", "INTC", "VZ", "T", "MRK",
"ABBV", "NKE", "MCD", "ADBE", "CRM", "ABT", "ORCL", "CMCSA", "LLY", "DHR",
"PM", "COST", "WFC", "IBM", "ACN", "TXN", "HON", "QCOM", "LIN", "BMY",
"CAT", "GS", "RTX", "LOW", "UPS", "AMGN", "MDT", "MS", "BKNG", "GE",
"UNP", "AXP", "C", "CVS", "BLK", "BA", "SPGI", "DE", "INTU", "PLD",
"SYK", "MO", "MDLZ", "LMT", "AMAT", "SCHW", "ISRG", "TMO", "ADI", "MMM",
"USB", "DUK", "PNC", "FDX", "SO", "CB", "REGN", "MMC", "CME", "ADI",
"APD", "BDX", "NOW", "GILD", "BK", "CL", "FIS", "EW", "ADP", "ZTS"]

def find_buys(tickers):
    for ticker in tickers:
        META = Price_Data(ticker)
        short, long = META.cal_macd()
        if META.stock_df["macd signal"][-1] > 0 and META.stock_df["macd signal"][-2] < 0:
            print(f"Buy {ticker}")
            plt.plot(short)
            plt.plot(long)
            plt.plot(META.stock_df["Close"])
            plt.show()

# find_buys(t)
"""Calculating the percentage of false positives for the two given moving averages (macd)"""

META = Price_Data("BEEM")
META.cal_macd()
META.stock_df.index = [i for i in range(len(META.stock_df["Close"]))]
closings = META.stock_df["Close"].pct_change()

macd = META.stock_df["macd signal"]
false_buy = (macd.shift(1) < 0) & (macd > 0) & (macd.shift(-1) < 0)
buy = (macd.shift(1) < 0) & (macd > 0)
buys = closings[buy]
test = [i+1 for i in closings[buy].index]
def find_false_sig(buys, test):
    true_pos = 0
    for i, t in enumerate(test):
        if test[i] < len(closings) and closings[t] > 0:
            true_pos += 1

    return f"True pos %: {true_pos/len(buys)}"

print(find_false_sig(buys, test))
