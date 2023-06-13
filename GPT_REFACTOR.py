# __init__.py
__all__ = ["Distributions", "Fitting", "Histogram", "Lifetime", "Parallelised", "Wrangling"]

def slt_help():
    print("Distributions: Lorentzian, Gaussian")
    print("Fitting: FitHistogram")
    print("Histogram: GetHistogramData, GetCenters, GetXrange, PlotHistogram, MeasureFWHM, PlotGamma, GammaEstimate")
    print("Lifetime: LifeTime, LifeTimeOverBins, StatisticalLifetime")
    print("Parallelised: Tau_Gaussian, Tau_Gaussian_Tuple, AQN_Generator, UnpackingStatLifeTimes, Tau_Gaussian_MP")
    print("Wrangling: Filter, Cutoff, AllowedQuantumNumbers, PlotLE")


# Dependencies.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
from quantiphy import Quantity
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import ast
from uncertainties import ufloat, ufloat_fromstr, unumpy
from scipy.interpolate import splrep, BSpline
from scipy.signal import savgol_filter


# Distributions.py
from .Dependencies import np

def lorentzian(x, x0, gamma):
    """
    Returns Lorentzian distribution

    Inputs:
        x: dependent variable values (list of floats)
        x0: mean (float)
        gamma: HWHM (float)

    Outputs:
        lorentz: Lorentzian distribution (list of floats)
    """
    numerator = (1 / np.pi) * (0.5 * gamma)
    denominator = ((x - x0) ** 2 + (0.5 * gamma) ** 2)

    lorentz = numerator / denominator

    return lorentz

def gaussian(x, mu, sigma):
    """
    Returns Gaussian distribution

    Inputs:
        x: dependent variable values (list of floats)
        mu: mean (float)
        sigma: standard deviation (float)

    Outputs:
        gauss: Gaussian distribution (list of floats)
    """
    gauss = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return gauss


# Fitting.py
from .Dependencies import curve_fit
from .Histogram import get_centers
from .Distributions import lorentzian

def fit_histogram(count, edges, formatted=True, guesses=None):
    """
    Fits Lorentzian function to histogram count and centers and extracts halfwidth

    Inputs:
        count: count for bin (np.1darray of floats)
        edges: bin edges (np.1darray of floats)
        formatted: format of lifetime (bool)
        guesses: initial parameter guesses (list of floats)

    Outputs:
        popt: fitted parameters (list of floats)
        lifetime: lifetime of state (value of float or str)
    """
    centers = get_centers(edges)

    if guesses is None:
        guesses = [1, 1]

    popt = curve_fit(lorentzian, centers, count, p0=guesses)[0]
    fwhm = popt[1]

    gamma = fwhm * 1.98630e-23
    hbar = 1.054571817e-34

    lifetime = hbar / gamma

    return [popt, lifetime] if formatted else [popt, fwhm, lifetime]


# Histogram.py
from .Dependencies import plt, savgol_filter

def get_histogram_data(data, bins, cutoffs=None):
    """
    Returns histogram data

    Inputs:
        data: data points (np.1darray of floats)
        bins: number of bins (int)
        cutoffs: histogram range (tuple of floats)

    Outputs:
        count: count for bin (np.1darray of floats)
        edges: bin edges (np.1darray of floats)
    """
    if cutoffs is None:
        cutoffs = [None, None]

    count, edges, _ = plt.hist(data, bins=bins, range=cutoffs, density=False, alpha=0.5)
    return count, edges

def get_centers(edges):
    """
    Returns centers of histogram bins

    Inputs:
        edges: bin edges (np.1darray of floats)

    Outputs:
        centers: centers of histogram bins (np.1darray of floats)
    """
    centers = (edges[:-1] + edges[1:]) / 2
    return centers

def get_xrange(data, x_range):
    """
    Returns x range based on data and specified range

    Inputs:
        data: data points (np.1darray of floats)
        x_range: specified x range (tuple of floats)

    Outputs:
        xmin: minimum x value (float)
        xmax: maximum x value (float)
    """
    if x_range is None:
        xmin, xmax = min(data), max(data)
    else:
        xmin, xmax = x_range

    return xmin, xmax

def plot_histogram(data, bins=100, cutoffs=None, x_range=None):
    """
    Plots histogram of data

    Inputs:
        data: data points (np.1darray of floats)
        bins: number of bins (int)
        cutoffs: histogram range (tuple of floats)
        x_range: specified x range (tuple of floats)
    """
    count, edges = get_histogram_data(data, bins, cutoffs)
    centers = get_centers(edges)
    xmin, xmax = get_xrange(data, x_range)

    plt.figure()
    plt.hist(data, bins=bins, range=cutoffs, density=False, alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.xlabel("Value")
    plt.ylabel("Count")


# Lifetime.py
from .Dependencies import np, unumpy, ufloat

def lifetime(data, peak_center, error=False, time_unit=None):
    """
    Calculates the lifetime of a peak center from time series data

    Inputs:
        data: time series data (list or array-like)
        peak_center: location of the peak center (float or Quantity)
        error: flag to compute uncertainty (bool)
        time_unit: time unit for uncertainty (str)

    Outputs:
        lifetime: calculated lifetime (float or Quantity)
        lifetime_uncertainty: calculated lifetime uncertainty (float or Quantity) if error=True, None otherwise
    """
    if isinstance(peak_center, Quantity):
        peak_center = peak_center.value

    time = np.arange(len(data))
    decay = data / np.max(data)

    # Fit exponential decay to the data
    popt, _ = curve_fit(lambda x, a, b: a * np.exp(-b * x), time, decay)
    decay_constant = popt[1]

    # Calculate lifetime
    lifetime = 1 / decay_constant

    if error:
        lifetime_uncertainty = 1 / (decay_constant ** 2) * np.sqrt(np.diag(np.linalg.inv(np.dot(np.transpose(jacobian(time, *popt)), jacobian(time, *popt)))))[1]
        if time_unit:
            lifetime_uncertainty *= time_unit

        return lifetime, lifetime_uncertainty
    else:
        return lifetime

def lifetime_over_bins(data, peak_centers, error=False, time_unit=None):
    """
    Calculates the lifetime of multiple peak centers from time series data

    Inputs:
        data: time series data (list or array-like)
        peak_centers: locations of the peak centers (list or array-like of floats or Quantities)
        error: flag to compute uncertainty (bool)
        time_unit: time unit for uncertainty (str)

    Outputs:
        lifetimes: calculated lifetimes (list of floats or Quantities)
        lifetime_uncertainties: calculated lifetime uncertainties (list of floats or Quantities) if error=True, None otherwise
    """
    lifetimes = []
    lifetime_uncertainties = []

    for center in peak_centers:
        lifetime, lifetime_uncertainty = lifetime(data, center, error, time_unit)
        lifetimes.append(lifetime)
        lifetime_uncertainties.append(lifetime_uncertainty)

    return lifetimes, lifetime_uncertainties

def statistical_lifetime(lifetimes, lifetime_uncertainties=None):
    """
    Calculates the statistical lifetime from a list of lifetimes

    Inputs:
        lifetimes: list of lifetimes (list or array-like of floats or Quantities)
        lifetime_uncertainties: list of lifetime uncertainties (list or array-like of floats or Quantities) or None (default)

    Outputs:
        statistical_lifetime: statistical lifetime (float or Quantity)
        statistical_lifetime_uncertainty: statistical lifetime uncertainty (float or Quantity) if lifetime_uncertainties is provided, None otherwise
    """
    if lifetime_uncertainties is not None:
        weights = 1 / np.array(lifetime_uncertainties) ** 2
        statistical_lifetime = np.average(lifetimes, weights=weights)
        statistical_lifetime_uncertainty = np.sqrt(1 / np.sum(weights))
        return statistical_lifetime, statistical_lifetime_uncertainty
    else:
        statistical_lifetime = np.mean(lifetimes)
        return statistical_lifetime


# Parallelised.py
from .Dependencies import ProcessPoolExecutor, partial, tqdm
from .Lifetime import lifetime

def tau_gaussian(data, n, error=False, time_unit=None):
    """
    Calculates the lifetime of multiple peaks from time series data using parallel processing

    Inputs:
        data: time series data (list or array-like)
        n: number of peaks to calculate the lifetime for (int)
        error: flag to compute uncertainty (bool)
        time_unit: time unit for uncertainty (str)

    Outputs:
        lifetimes: calculated lifetimes (list of floats or Quantities)
        lifetime_uncertainties: calculated lifetime uncertainties (list of floats or Quantities) if error=True, None otherwise
    """
    peak_centers = []

    for i in range(n):
        peak_centers.append(2 * i + 1)

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(partial(lifetime, data, error=error, time_unit=time_unit), peak_centers), total=n))

    lifetimes = [result[0] for result in results]
    lifetime_uncertainties = [result[1] for result in results] if error else None

    return lifetimes, lifetime_uncertainties

def tau_gaussian_tuple(data, n, error=False, time_unit=None):
    """
    Calculates the lifetime of multiple peaks from time series data using parallel processing and returns as a tuple

    Inputs:
        data: time series data (list or array-like)
        n: number of peaks to calculate the lifetime for (int)
        error: flag to compute uncertainty (bool)
        time_unit: time unit for uncertainty (str)

    Outputs:
        results: tuple of calculated lifetimes (list of floats or Quantities) and lifetime uncertainties (list of floats or Quantities) if error=True, None otherwise
    """
    lifetimes, lifetime_uncertainties = tau_gaussian(data, n, error, time_unit)
    results = (lifetimes, lifetime_uncertainties)
    return results

def aqn_generator(data, n, error=False, time_unit=None):
    """
    Calculates the lifetime of multiple peaks from time series data using parallel processing and returns AQN-compatible output

    Inputs:
        data: time series data (list or array-like)
        n: number of peaks to calculate the lifetime for (int)
        error: flag to compute uncertainty (bool)
        time_unit: time unit for uncertainty (str)

    Outputs:
        aqn_output: AQN-compatible output (dict)
    """
    lifetimes, lifetime_uncertainties = tau_gaussian(data, n, error, time_unit)

    aqn_output = {"n_peaks": n, "lifetimes": lifetimes}

    if error:
        aqn_output["lifetime_uncertainties"] = lifetime_uncertainties

    return aqn_output

def unpacking_stat_life_times(data, n, error=False, time_unit=None):
    """
    Calculates the lifetime of multiple peaks from time series data using parallel processing and returns AQN-compatible statistical lifetime output

    Inputs:
        data: time series data (list or array-like)
        n: number of peaks to calculate the lifetime for (int)
        error: flag to compute uncertainty (bool)
        time_unit: time unit for uncertainty (str)

    Outputs:
        statistical_lifetime_output: AQN-compatible statistical lifetime output (dict)
    """
    lifetimes, lifetime_uncertainties = tau_gaussian(data, n, error, time_unit)
    statistical_lifetime_val, statistical_lifetime_unc = statistical_lifetime(lifetimes, lifetime_uncertainties)

    statistical_lifetime_output = {"n_peaks": n, "lifetimes": lifetimes, "statistical_lifetime": statistical_lifetime_val}

    if error:
        statistical_lifetime_output["lifetime_uncertainties"] = lifetime_uncertainties
        statistical_lifetime_output["statistical_lifetime_uncertainty"] = statistical_lifetime_unc

    return statistical_lifetime_output

def tau_gaussian_mp(data, n, error=False, time_unit=None):
    """
    Calculates the lifetime of multiple peaks from time series data using parallel processing (multiprocessing) and returns AQN-compatible output

    Inputs:
        data: time series data (list or array-like)
        n: number of peaks to calculate the lifetime for (int)
        error: flag to compute uncertainty (bool)
        time_unit: time unit for uncertainty (str)

    Outputs:
        aqn_output: AQN-compatible output (dict)
    """
    lifetimes, lifetime_uncertainties = tau_gaussian_tuple(data, n, error, time_unit)

    aqn_output = {"n_peaks": n, "lifetimes": lifetimes}

    if error:
        aqn_output["lifetime_uncertainties"] = lifetime_uncertainties

    return aqn_output


# Wrangling.py
from .Dependencies import np, pd, plt, interpolate

def filter(data, cutoffs):
    """
    Filters data based on cutoffs

    Inputs:
        data: data points (np.1darray of floats)
        cutoffs: cutoff range (tuple of floats)

    Outputs:
        filtered_data: filtered data points (np.1darray of floats)
    """
    filtered_data = np.array([x for x in data if cutoffs[0] <= x <= cutoffs[1]])
    return filtered_data

def cutoff(data, cutoffs):
    """
    Applies cutoff to data based on cutoffs

    Inputs:
        data: data points (np.1darray of floats)
        cutoffs: cutoff range (tuple of floats)

    Outputs:
        cutoff_data: data points with cutoff applied (np.1darray of floats)
    """
    cutoff_data = np.where((cutoffs[0] <= data) & (data <= cutoffs[1]), data, np.nan)
    return cutoff_data

def allowed_quantum_numbers(data, cutoffs, bins, error=False):
    """
    Determines the allowed quantum numbers from the data

    Inputs:
        data: data points (np.1darray of floats)
        cutoffs: cutoff range (tuple of floats)
        bins: number of bins (int)
        error: flag to compute uncertainty (bool)

    Outputs:
        quantum_numbers: allowed quantum numbers (list of floats)
        quantum_number_uncertainties: quantum number uncertainties (list of floats) if error=True, None otherwise
    """
    filtered_data = filter(data, cutoffs)
    counts, edges = np.histogram(filtered_data, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    peaks, _ = find_peaks(counts, prominence=(np.max(counts) * 0.1, None))
    quantum_numbers = centers[peaks]

    if error:
        quantum_number_uncertainties = []

        for i in range(len(peaks)):
            # Calculate uncertainties as the width at half-maximum (FWHM) of each peak
            peak_counts = counts[peaks[i]]
            peak_center = centers[peaks[i]]
            left_idx = np.where(counts[:peaks[i]] <= peak_counts / 2)[0][-1]
            right_idx = np.where(counts[peaks[i]:] <= peak_counts / 2)[0][0] + peaks[i]

            fwhm = centers[right_idx] - centers[left_idx]
            quantum_number_uncertainty = fwhm / 2.355  # Assuming Gaussian distribution

            quantum_number_uncertainties.append(quantum_number_uncertainty)

        return quantum_numbers, quantum_number_uncertainties
    else:
        return quantum_numbers

def plot_le(data, cutoffs, bins, error=False):
    """
    Plots the level scheme of the data

    Inputs:
        data: data points (np.1darray of floats)
        cutoffs: cutoff range (tuple of floats)
        bins: number of bins (int)
        error: flag to compute uncertainty (bool)
    """
    quantum_numbers, quantum_number_uncertainties = allowed_quantum_numbers(data, cutoffs, bins, error)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for i, num in enumerate(quantum_numbers):
        ax.annotate(f"{num}", (0, i), fontsize=12, ha="right", va="center")

    plt.ylim(-0.5, len(quantum_numbers) - 0.5)
    plt.show()


# Example usage:
if __name__ == "__main__":
    data = np.random.normal(0, 1, 1000)
    bins = 20
    cutoffs = (-2, 2)

    filtered_data = filter(data, cutoffs)
    cutoff_data = cutoff(data, cutoffs)
    quantum_numbers, quantum_number_uncertainties = allowed_quantum_numbers(data, cutoffs, bins, error=True)

    plot_le(data, cutoffs, bins, error=True)
    plot_le(data, cutoffs, bins, error=False)
