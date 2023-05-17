# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from quantiphy      import Quantity
from scipy.optimize import curve_fit
from tqdm           import tqdm

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import ast
from uncertainties import ufloat

#https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
from scipy.interpolate import splrep, BSpline
#https://python.plainenglish.io/my-favorite-way-to-smooth-noisy-data-with-python-bd28abe4b7d0
from scipy.signal import savgol_filter

from .Distributions import *
from .Fitting       import *
from .Histogram     import *
from .Lifetime      import *
from .Parallelised  import *
from .Wrangling     import *
# %%
