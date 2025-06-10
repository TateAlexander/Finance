import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import geom

#  Attempting to find the optimal amount of cash to take out of the investment each day for maximal profit. Seems to be either 0 or 100% depending on if the p is low enough to make the gain/loss ratio profitable.

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

optimal = sorted(multiSim(conserv_perc), key=lambda x:x[1])
print(optimal)
plt.plot(optimal)
plt.show()
