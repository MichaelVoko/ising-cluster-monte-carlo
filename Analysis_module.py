"""
Analysis module
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import Algorithm_module as alg
import Simulation_module as sim


"""
Autocorrelation time
"""


def sample_autocovariance(x, tmax):
    """Compute the autocorrelation of the time series x for t = 0,1,...,tmax-1."""
    x_shifted = x - np.mean(x)
    covariances = np.array(
        [np.dot(x_shifted[: len(x) - t], x_shifted[t:]) / len(x) for t in range(tmax)]
    )
    normalised = covariances / covariances[0]
    return normalised


def correlation_time(
    autocov,
):  # NB this function was not actually used in the data in report; the exponential fit method was preferred
    """Return the index of the first entry that is smaller than
    autocov[0]/e or the length of autocov if none are smaller."""
    smaller = np.where(autocov < np.exp(-1) * autocov[0])[0]
    return smaller[0] if len(smaller) > 0 else len(autocov)


def find_correlation_time(measured_taus):  # Neither was this
    """Compute mean τ_int and its standard error from a list of runs."""
    taus = np.asarray(measured_taus)
    n = len(taus)
    mean = np.mean(taus)
    stderr = np.std(taus) / np.sqrt(n)
    return mean, stderr


def exponential(t, tau):
    A = 1
    return A * np.exp(-t / tau)


def find_autocorrelation_time(series, fit=True):
    if fit:
        fit_range = 200  # use early values to avoid noise
        popt, _ = curve_fit(
            exponential, np.arange(0, len(series), 1)[:fit_range], series[:fit_range]
        )
        tau = popt[0]
    else:
        tau = correlation_time(series)

    return tau


def plot_autocorr(betaJ, algo, series, errors, spacing):
    tau_fit = find_autocorrelation_time(series)
    fit_curve = exponential(np.arange(0, len(series), 1), tau_fit)

    plt.plot(
        np.arange(0, len(series), 1),
        fit_curve,
        "r-",
        label=f"Exponential Fit: $\\tau$ = {tau_fit:.2f}",
    )

    plt.errorbar(
        np.arange(0, len(series), spacing),
        series[::spacing],
        yerr=errors[::spacing],
        fmt="o-",
        capsize=1,
        label="Metropolis",
        color="blue",
        linewidth=0.5,
    )
    plt.xlabel("Time lag (in sweeps)")
    plt.ylabel("Autocorrelation")
    plt.title(
        f"Autocorrelation Decay of {algo.__name__} Algorithm with Exponential Fit ($\\beta J = {betaJ}$)"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


""" 
Time per step 
"""


def find_time_per_step(times, steps, num_simulations=5):
    """Run a single simulation and return the time per step."""

    # Calculate total time and time per step
    times_per_step = np.array(times) / steps

    time_per_step_mean = np.mean(times_per_step)
    time_per_step_error = np.std(np.array(times_per_step)) / np.sqrt(num_simulations)

    return time_per_step_mean, time_per_step_error


def find_rate_or_size(rates, num_simulations=5):

    rates_mean = np.mean(rates)
    rates_error = np.std(np.array(rates)) / np.sqrt(num_simulations)

    return rates_mean, rates_error
