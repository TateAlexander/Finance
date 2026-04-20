import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from scipy.stats import linregress, t, gaussian_kde


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
START_DATE = "2025-10-01"
END_DATE = "2026-04-20"

INV_ETFS = {
    "XLE": START_DATE,
    "TQQQ": START_DATE,
    "SQQQ": START_DATE,
}


# ------------------------------------------------------------
# Interval breaking by slope t-test
# ------------------------------------------------------------
def slope_confidence_interval(y, x=None, alpha=0.10):
    """
    Compute slope, intercept, stderr, and a t-based confidence interval for slope.
    """
    y = np.asarray(y, dtype=float)
    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.asarray(x, dtype=float)

    if len(y) < 3:
        raise ValueError("Need at least 3 points for a slope t interval.")

    res = linregress(x, y)
    slope = res.slope
    intercept = res.intercept
    slope_stderr = res.stderr if res.stderr is not None else 0.0

    t_crit = t.ppf(1 - alpha / 2, len(y) - 2)
    conf = (
        slope - t_crit * slope_stderr,
        slope + t_crit * slope_stderr
    )
    return slope, intercept, slope_stderr, conf


def find_break(closes, width=5, alpha=0.10, min_interval_len=10):
    """
    Break the series into intervals using the original idea:
    compare each window's slope against the PRIOR window's slope CI.

    Returns:
        dict[str, pd.Series]
    """
    closes = closes.dropna().astype(float).copy()
    n = len(closes)

    if n < max(width, 3):
        return {"interval 1": closes}

    intervals = {}
    interval_count = 1
    current_start = 0
    prev_conf = None

    for start in range(0, n, width):
        stop = min(start + width, n)
        window = closes.iloc[start:stop]

        if len(window) < 3:
            continue

        x = np.arange(start, stop, dtype=float)
        slope, intercept, slope_stderr, conf = slope_confidence_interval(
            window.values, x=x, alpha=alpha
        )

        if prev_conf is None:
            prev_conf = conf
            continue

        # Break if current slope lies outside the prior slope's CI
        is_break = (slope < prev_conf[0]) or (slope > prev_conf[1])

        if is_break and (start - current_start) >= min_interval_len:
            intervals[f"interval {interval_count}"] = closes.iloc[current_start:start].copy()
            interval_count += 1
            current_start = start

        prev_conf = conf

    # Final interval
    final_interval = closes.iloc[current_start:].copy()
    if len(final_interval) > 0:
        intervals[f"interval {interval_count}"] = final_interval

    return intervals


# ------------------------------------------------------------
# Mean-reversion-to-regression-line model
# ------------------------------------------------------------
class MeanReversionLineModel:
    """
    Model:
        ΔP_t = k (reg_t - P_{t-1}) + ε_t

    so that
        P_t = P_{t-1} + k (reg_t - P_{t-1}) + ε_t
    """

    def __init__(self, ticker, closes, use_log_price=False):
        closes = pd.Series(closes).dropna().astype(float)

        if len(closes) < 3:
            raise ValueError("Need at least 3 prices to fit the model.")

        self.ticker = ticker
        self.use_log_price = use_log_price
        self.prices = closes.copy()
        self.index = closes.index
        self.n = len(closes)
        self.x = np.arange(self.n, dtype=float)

        if use_log_price:
            self.model_prices = np.log(self.prices.values)
        else:
            self.model_prices = self.prices.values.copy()

        self.is_fitted = False

    def compute_reg_line(self):
        m, b = np.polyfit(self.x, self.model_prices, 1)
        self.slope = float(m)
        self.intercept = float(b)

        self.reg_line_model = self.slope * self.x + self.intercept

        if self.use_log_price:
            reg_price = np.exp(self.reg_line_model)
        else:
            reg_price = self.reg_line_model.copy()

        self.reg_line = pd.Series(reg_price, index=self.index, name="reg_line")
        return self.reg_line

    def estimate_k(self, clip=(0.0, 2.0)):
        """
        Estimate k from:
            delta_t = k * gap_t + eps_t

        where
            delta_t = P_t - P_{t-1}
            gap_t   = reg_t - P_{t-1}
        """
        if not hasattr(self, "reg_line_model"):
            self.compute_reg_line()

        delta = self.model_prices[1:] - self.model_prices[:-1]
        gap = self.reg_line_model[1:] - self.model_prices[:-1]

        denom = np.dot(gap, gap)
        if denom < 1e-12:
            k = 0.0
        else:
            k = np.dot(gap, delta) / denom

        if clip is not None:
            k = float(np.clip(k, clip[0], clip[1]))

        self.k_est = k
        self.innovations = delta - self.k_est * gap
        return self.k_est

    def fit_noise(self):
        """
        Fit a KDE to the innovations ε_t.
        """
        if not hasattr(self, "innovations"):
            self.estimate_k()

        innovations = np.asarray(self.innovations, dtype=float)

        if len(innovations) < 2 or np.std(innovations) < 1e-10:
            self.noise_kde = None
            self.noise_std = float(np.std(innovations)) if len(innovations) > 0 else 0.0
        else:
            self.noise_kde = gaussian_kde(innovations, bw_method="scott")
            self.noise_std = float(np.std(innovations))

    def fit(self):
        self.compute_reg_line()
        self.estimate_k()
        self.fit_noise()
        self.is_fitted = True
        return self

    def sample_innovations(self, size):
        if self.noise_kde is not None:
            return self.noise_kde.resample(size).ravel()

        std = max(self.noise_std, 1e-8)
        return np.random.normal(loc=0.0, scale=std, size=size)

    def simulate_paths(self, n_paths=5000):
        """
        Simulate price paths over the fitted interval.
        """
        if not self.is_fitted:
            self.fit()

        sims = np.zeros((n_paths, self.n), dtype=float)
        sims[:, 0] = self.model_prices[0]

        for p in range(n_paths):
            eps = self.sample_innovations(self.n - 1)
            for i in range(1, self.n):
                prev = sims[p, i - 1]
                reg_i = self.reg_line_model[i]
                sims[p, i] = prev + self.k_est * (reg_i - prev) + eps[i - 1]

        if self.use_log_price:
            sims = np.exp(sims)

        return sims

    def residual_paths(self, n_paths=5000):
        """
        Simulated residuals relative to the regression line, in PRICE space.
        """
        sims = self.simulate_paths(n_paths=n_paths)
        reg = self.reg_line.values.reshape(1, -1)
        return sims - reg

    def residual_bands(self, alpha=0.10, n_paths=5000):
        """
        Pointwise simulation bands around the regression line.
        """
        resids = self.residual_paths(n_paths=n_paths)

        lower_resid = np.quantile(resids, alpha / 2, axis=0)
        upper_resid = np.quantile(resids, 1 - alpha / 2, axis=0)

        lower = pd.Series(self.reg_line.values + lower_resid, index=self.index, name="lower")
        upper = pd.Series(self.reg_line.values + upper_resid, index=self.index, name="upper")
        return lower, upper

    def buy_sell_levels(self, alpha=0.10, n_paths=5000):
        """
        Suggested buy/sell levels at the LAST point of the interval.
        """
        resids = self.residual_paths(n_paths=n_paths)
        last_resids = resids[:, -1]

        buy_level = self.reg_line.iloc[-1] + np.quantile(last_resids, alpha / 2)
        sell_level = self.reg_line.iloc[-1] + np.quantile(last_resids, 1 - alpha / 2)

        return float(buy_level), float(sell_level)

    def plot(self, alpha=0.10, n_paths=5000, title=None):
        lower, upper = self.residual_bands(alpha=alpha, n_paths=n_paths)

        plt.figure(figsize=(12, 6))
        plt.plot(self.index, self.prices.values, label="Close", linewidth=1.7)
        plt.plot(self.index, self.reg_line.values, label="Regression line", linewidth=2.0)
        plt.plot(self.index, lower.values, label=f"Lower {100*(alpha/2):.1f}% band", linestyle="--")
        plt.plot(self.index, upper.values, label=f"Upper {100*(1-alpha/2):.1f}% band", linestyle="--")
        plt.title(title or f"{self.ticker} last interval mean-reversion fit")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------
# Analysis helpers
# ------------------------------------------------------------
def analyze_last_interval(
    ticker,
    start_date,
    end_date,
    width=5,
    break_alpha=0.10,
    min_interval_len=10,
    sim_alpha=0.10,
    n_paths=5000,
    use_log_price=False,
):
    closes = yf.Ticker(ticker).history(
        start=start_date,
        end=end_date,
        interval="1d"
    )["Close"].dropna()

    if len(closes) < 10:
        print(f"{ticker}: not enough data.")
        return None

    intervals = find_break(
        closes,
        width=width,
        alpha=break_alpha,
        min_interval_len=min_interval_len
    )

    interval_name, last_interval = list(intervals.items())[-1]

    if len(last_interval) < 3:
        print(f"{ticker}: last interval too short.")
        return None

    model = MeanReversionLineModel(
        ticker=ticker,
        closes=last_interval,
        use_log_price=use_log_price
    ).fit()

    buy_level, sell_level = model.buy_sell_levels(alpha=sim_alpha, n_paths=n_paths)

    print(f"\n{ticker} — {interval_name}")
    print(f"Bars in last interval: {len(last_interval)}")
    print(f"Regression slope:      {model.slope:.6f}")
    print(f"Mean-reversion k:      {model.k_est:.6f}")
    print(f"Suggested buy level:   {buy_level:.4f}")
    print(f"Suggested sell level:  {sell_level:.4f}")

    model.plot(
        alpha=sim_alpha,
        n_paths=n_paths,
        title=f"{ticker} — {interval_name}"
    )

    return {
        "ticker": ticker,
        "all_closes": closes,
        "intervals": intervals,
        "last_interval_name": interval_name,
        "last_interval": last_interval,
        "model": model,
        "buy_level": buy_level,
        "sell_level": sell_level,
    }


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    results = {}

    for ticker, start in INV_ETFS.items():
        result = analyze_last_interval(
            ticker=ticker,
            start_date=start,
            end_date=END_DATE,
            width=5,
            break_alpha=0.10,
            min_interval_len=10,
            sim_alpha=0.10,
            n_paths=3000,
            use_log_price=False,   # set True if you want straight lines in log-price space
        )
        results[ticker] = result

#-----------------------------------------------------------------------------------------------------------------------
