import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from scipy.stats import linregress, t, gaussian_kde


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
START_DATE = "2026-02-01"
END_DATE = "2026-07-10"

INV_ETFS = {
    "AEP": START_DATE,
    "AM": START_DATE,
}


# ------------------------------------------------------------
# Interval breaking by slope t-test
# ------------------------------------------------------------
def slope_confidence_interval(y, x=None, alpha=0.10):
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
            window.values,
            x=x,
            alpha=alpha
        )

        if prev_conf is None:
            prev_conf = conf
            continue

        is_break = (slope < prev_conf[0]) or (slope > prev_conf[1])

        if is_break and (start - current_start) >= min_interval_len:
            intervals[f"interval {interval_count}"] = closes.iloc[current_start:start].copy()
            interval_count += 1
            current_start = start

        prev_conf = conf

    final_interval = closes.iloc[current_start:].copy()

    if len(final_interval) > 0:
        intervals[f"interval {interval_count}"] = final_interval

    return intervals


# ------------------------------------------------------------
# Mean-reversion-to-regression-line model
# ------------------------------------------------------------
class MeanReversionLineModel:
    """
    Main model:

        Price_t = LinearTrend_t + Residual_t

    Original simulation model:

        ΔP_t = k(reg_t - P_{t-1}) + ε_t

    Hypothesis test added:

        Residual_t = Price_t - LinearTrend_t

        Residual_{t+1} = phi * Residual_t + noise

    If 0 < phi < 1, residuals are mean reverting.

        OU k = -log(phi)
        half-life = log(2) / k
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
        if not hasattr(self, "innovations"):
            self.estimate_k()

        innovations = np.asarray(self.innovations, dtype=float)

        if len(innovations) < 2 or np.std(innovations) < 1e-10:
            self.noise_kde = None
            self.noise_std = float(np.std(innovations)) if len(innovations) > 0 else 0.0
        else:
            self.noise_kde = gaussian_kde(innovations, bw_method="scott")
            self.noise_std = float(np.std(innovations))

    def estimate_residual_mean_reversion(self):
        """
        Tests whether residuals around the fitted linear trend are mean reverting.

        Residual:
            X_t = Price_t - Trend_t

        AR(1):
            X_{t+1} = phi * X_t + eps_t

        OU conversion:
            k = -log(phi)

        Unit-root-style regression:
            ΔX_t = beta * X_t + eps_t

        If beta < 0, residuals tend to move back toward zero.
        """

        if not hasattr(self, "reg_line_model"):
            self.compute_reg_line()

        residuals = self.model_prices - self.reg_line_model

        x_t = residuals[:-1]
        x_next = residuals[1:]
        dx = x_next - x_t

        denom = np.dot(x_t, x_t)

        if denom < 1e-12:
            phi = np.nan
            ou_k = np.nan
            half_life = np.nan
        else:
            phi = np.dot(x_t, x_next) / denom

            if 0 < phi < 1:
                ou_k = -np.log(phi)
                half_life = np.log(2) / ou_k
            else:
                ou_k = np.nan
                half_life = np.nan

        if len(x_t) >= 3 and np.std(x_t) > 1e-12:
            mr_test = linregress(x_t, dx)

            beta = mr_test.slope
            beta_pvalue = mr_test.pvalue
            beta_tstat = mr_test.slope / mr_test.stderr if mr_test.stderr not in [None, 0] else np.nan
            r_squared = mr_test.rvalue ** 2
        else:
            beta = np.nan
            beta_pvalue = np.nan
            beta_tstat = np.nan
            r_squared = np.nan

        self.trend_residuals = pd.Series(
            residuals,
            index=self.index,
            name="trend_residual"
        )

        self.resid_phi = float(phi) if not np.isnan(phi) else np.nan
        self.resid_ou_k = float(ou_k) if not np.isnan(ou_k) else np.nan
        self.resid_half_life = float(half_life) if not np.isnan(half_life) else np.nan
        self.resid_beta = float(beta) if not np.isnan(beta) else np.nan
        self.resid_beta_tstat = float(beta_tstat) if not np.isnan(beta_tstat) else np.nan
        self.resid_beta_pvalue = float(beta_pvalue) if not np.isnan(beta_pvalue) else np.nan
        self.resid_r_squared = float(r_squared) if not np.isnan(r_squared) else np.nan

        if not np.isnan(phi):
            if 0 < phi < 1:
                conclusion = "Residuals show mean reversion around the linear trend."
            elif phi >= 1:
                conclusion = "Residuals do NOT show mean reversion; phi is >= 1."
            else:
                conclusion = "Residuals alternate sign strongly; standard OU interpretation is unstable."
        else:
            conclusion = "Residual mean reversion could not be estimated."

        self.resid_mr_stats = {
            "phi": self.resid_phi,
            "ou_k": self.resid_ou_k,
            "half_life_bars": self.resid_half_life,
            "beta_delta_regression": self.resid_beta,
            "beta_tstat": self.resid_beta_tstat,
            "beta_pvalue": self.resid_beta_pvalue,
            "r_squared": self.resid_r_squared,
            "residual_std": float(np.std(residuals)),
            "conclusion": conclusion
        }

        return self.resid_mr_stats

    def fit(self):
        self.compute_reg_line()
        self.estimate_k()
        self.fit_noise()
        self.estimate_residual_mean_reversion()
        self.is_fitted = True
        return self

    def sample_innovations(self, size):
        if self.noise_kde is not None:
            return self.noise_kde.resample(size).ravel()

        std = max(self.noise_std, 1e-8)

        return np.random.normal(loc=0.0, scale=std, size=size)

    def simulate_paths(self, n_paths=5000):
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
        sims = self.simulate_paths(n_paths=n_paths)
        reg = self.reg_line.values.reshape(1, -1)

        return sims - reg

    def residual_bands(self, alpha=0.10, n_paths=5000):
        resids = self.residual_paths(n_paths=n_paths)

        lower_resid = np.quantile(resids, alpha / 2, axis=0)
        upper_resid = np.quantile(resids, 1 - alpha / 2, axis=0)

        lower = pd.Series(
            self.reg_line.values + lower_resid,
            index=self.index,
            name="lower"
        )

        upper = pd.Series(
            self.reg_line.values + upper_resid,
            index=self.index,
            name="upper"
        )

        return lower, upper

    def buy_sell_levels(self, alpha=0.10, n_paths=5000):
        resids = self.residual_paths(n_paths=n_paths)
        last_resids = resids[:, -1]

        buy_level = self.reg_line.iloc[-1] + np.quantile(last_resids, alpha / 2)
        sell_level = self.reg_line.iloc[-1] + np.quantile(last_resids, 1 - alpha / 2)

        return float(buy_level), float(sell_level)

    def plot(self, alpha=0.10, n_paths=5000, title=None):
        lower, upper = self.residual_bands(alpha=alpha, n_paths=n_paths)

        plt.figure(figsize=(12, 6))

        plt.plot(
            self.index,
            self.prices.values,
            label="Close",
            linewidth=1.7
        )

        plt.plot(
            self.index,
            self.reg_line.values,
            label="Regression line",
            linewidth=2.0
        )

        plt.plot(
            self.index,
            lower.values,
            label=f"Lower {100*(alpha/2):.1f}% band",
            linestyle="--"
        )

        plt.plot(
            self.index,
            upper.values,
            label=f"Upper {100*(1-alpha/2):.1f}% band",
            linestyle="--"
        )

        plt.title(title or f"{self.ticker} last interval mean-reversion fit")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_residuals(self, title=None):
        if not hasattr(self, "trend_residuals"):
            self.estimate_residual_mean_reversion()

        plt.figure(figsize=(12, 4))

        plt.plot(
            self.trend_residuals.index,
            self.trend_residuals.values,
            label="Residual = price - trend",
            linewidth=1.7
        )

        plt.axhline(0, linestyle="--", linewidth=1.2)

        plt.title(title or f"{self.ticker} residuals around linear trend")
        plt.xlabel("Date")
        plt.ylabel("Residual")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_residual_scatter(self, title=None):
        if not hasattr(self, "trend_residuals"):
            self.estimate_residual_mean_reversion()

        residuals = self.trend_residuals.values

        x_t = residuals[:-1]
        x_next = residuals[1:]

        plt.figure(figsize=(6, 6))

        plt.scatter(x_t, x_next, alpha=0.7)

        if len(x_t) >= 3 and np.std(x_t) > 1e-12:
            fit = linregress(x_t, x_next)
            xs = np.linspace(np.min(x_t), np.max(x_t), 100)
            ys = fit.intercept + fit.slope * xs

            plt.plot(xs, ys, label=f"AR(1) fit: phi={fit.slope:.3f}")

        plt.axhline(0, linestyle="--", linewidth=1.0)
        plt.axvline(0, linestyle="--", linewidth=1.0)

        plt.title(title or f"{self.ticker} residual AR(1) test")
        plt.xlabel("Residual at t")
        plt.ylabel("Residual at t+1")
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

    buy_level, sell_level = model.buy_sell_levels(
        alpha=sim_alpha,
        n_paths=n_paths
    )

    mr = model.resid_mr_stats

    print(f"\n{ticker} — {interval_name}")
    print(f"Bars in last interval:        {len(last_interval)}")
    print(f"Regression slope:             {model.slope:.6f}")
    print(f"Original model k:             {model.k_est:.6f}")
    print(f"Suggested buy level:          {buy_level:.4f}")
    print(f"Suggested sell level:         {sell_level:.4f}")

    print("\n--- Hypothesis Test: Linear Trend + Mean-Reverting Residual ---")
    print(f"Residual AR(1) phi:           {mr['phi']:.6f}")
    print(f"Residual OU/Vasicek k:        {mr['ou_k']:.6f}")
    print(f"Residual half-life:           {mr['half_life_bars']:.2f} bars")
    print(f"Delta-regression beta:        {mr['beta_delta_regression']:.6f}")
    print(f"Beta t-stat:                  {mr['beta_tstat']:.6f}")
    print(f"Beta p-value:                 {mr['beta_pvalue']:.6f}")
    print(f"Residual AR relationship R^2: {mr['r_squared']:.4f}")
    print(f"Residual std:                 {mr['residual_std']:.6f}")
    print(f"Conclusion:                   {mr['conclusion']}")

    model.plot(
        alpha=sim_alpha,
        n_paths=n_paths,
        title=f"{ticker} — {interval_name}"
    )

    model.plot_residuals(
        title=f"{ticker} — residuals around linear trend"
    )

    model.plot_residual_scatter(
        title=f"{ticker} — residual AR(1) test"
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
        "residual_mean_reversion_stats": mr,
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
            use_log_price=False,
        )

        results[ticker] = result
