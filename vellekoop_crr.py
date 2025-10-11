import numpy as np
from scipy.interpolate import PchipInterpolator
from collections import defaultdict


def map_dividends_to_steps(dt, N, dividends=None):
    """
    Map continuous dividend times to discrete binomial tree steps.

    Parameters
    ----------
    dt : float
        Time step size (T / N).
    N : int
        Total number of time steps.
    dividends : list of (float, float), optional
        List of (time, amount) for discrete cash dividends.
        If None or empty, no dividends are applied.

    Returns
    -------
    dict
        Mapping {step_index: total_dividend} where step_index is the
        integer time step at or just after the dividend date. If multiple
        dividends fall on the same step, amounts are summed.
    """
    
    dividend_steps = defaultdict(float)

    if not dividends:
        return {}

    for t, D in dividends:
        if D <= 0:
            continue
        step = int(np.floor(t / dt))
        if step <= N:   # ignore dividends beyond maturity
            dividend_steps[step] += D

    return dict(dividend_steps)
            

def interpolate_post_dividend(S_nodes, values, D):
    """
    Apply Vellekoop dividend adjustment by interpolating option values 
    after a discrete cash dividend.

    Parameters
    ----------
    S_nodes : array_like
        Stock prices at the current tree step (before dividend).
    values : array_like
        Option values corresponding to S_nodes at time t+.
    D : float
        Dividend amount to subtract from stock prices.

    Returns
    -------
    ndarray
        Adjusted option values at time t- corresponding to original S_nodes.

    Notes
    -----
    Implements:
        V(t-, S-) = V(t+, max(S- - D, 0))
    Uses monotone PCHIP interpolation to evaluate values at shifted stock 
    prices S- - D. Negative stock levels are floored at zero.
    """
    S_nodes = np.asarray(S_nodes, dtype=float)
    values = np.asarray(values, dtype=float)

    if S_nodes.shape != values.shape:
        raise ValueError("S_nodes and values must have the same shape.")

    # Shift stock levels after dividend
    S_after = np.maximum(S_nodes - D, 0.0)

    # Interpolate option values at shifted stock levels
    cs = PchipInterpolator(S_nodes, values, extrapolate=True)
    adjusted_values = cs(S_after)

    return adjusted_values


def american_option_crr_binomial_with_cash_dividends(S0, K, r, sigma, T, N, dividends, option_type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    # terminal stock prices
    stockGrid = S0 * u**np.arange(-N, N+1, 2)

    # option payoff at maturity
    values = np.maximum(stockGrid - K, 0.0) if option_type == "call" else np.maximum(K - stockGrid, 0.0)

    # convert dividend times to step indices
    dividend_steps = map_dividends_to_steps(dt, N, dividends)
    
    # backward induction
    for i in range(N-1, -1, -1):
        # standard backward step
        values = disc * (p * values[1:] + (1-p) * values[:-1])

        # stock prices at this time step
        stockGrid[: i + 1] /= d

        # dividend adjustment (if a dividend occurs here)
        if i in dividend_steps:
            D = dividend_steps[i]
            values = interpolate_post_dividend(stockGrid[: i + 1], values, D)
            
        values = np.maximum(values, stockGrid[: i + 1] - K) if option_type == "call" else np.maximum(values, K - stockGrid[: i + 1])

    return values[0]




S0 = 100.0
T = 1.0
r = 0.05
sigma = 0.3
N = 1000

ts = [0.1, 0.5, 0.9]
Ks = [70.0, 100.0, 130.0]
prcs = []
for t in ts:
    div_times = np.array([t])
    div_amounts = np.repeat(7.0, len(div_times))
    dividends = [(t, d) for t,d in zip(div_times, div_amounts)]
    for K in Ks:
        V0 = american_option_crr_binomial_with_cash_dividends(S0, K, r, sigma, T, N, dividends, "call")
        prcs.append(V0)





