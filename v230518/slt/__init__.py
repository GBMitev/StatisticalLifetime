# # %%
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.stats as stats

# from quantiphy      import Quantity
# from scipy.optimize import curve_fit
# from tqdm           import tqdm

# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
# import ast
# from uncertainties import ufloat

# #https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
# from scipy.interpolate import splrep, BSpline
# #https://python.plainenglish.io/my-favorite-way-to-smooth-noisy-data-with-python-bd28abe4b7d0
# from scipy.signal import savgol_filter

from .Distributions import Lorentzian,Gaussian
from .Fitting       import FitHistogram
from .Histogram     import GetHistogramData ,GetCenters ,GetXrange ,PlotHistogram ,MeasureFWHM ,PlotGamma ,GammaEstimate
from .Lifetime      import LifeTime, LifeTimeOverBins, StatisticalLifetime
from .Parallelised  import Tau_Gaussian,Tau_Gaussian_Tuple,AQN_Generator,UnpackingStatLifeTimes,Tau_Gaussian_MP
from .Wrangling     import Filter, Cutoff, AllowedQuantumNumbers, PlotLE
# # %%
