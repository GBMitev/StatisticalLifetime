# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from quantiphy      import Quantity
from scipy.optimize import curve_fit
from tqdm           import tqdm
from scipy          import interpolate


from concurrent.futures import ProcessPoolExecutor
from functools import partial
import ast
from uncertainties import ufloat