# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy          import interpolate
from quantiphy      import Quantity
from scipy.optimize import curve_fit
from tqdm           import tqdm

#https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
from scipy.interpolate import splrep, BSpline
#https://python.plainenglish.io/my-favorite-way-to-smooth-noisy-data-with-python-bd28abe4b7d0
from scipy.signal import savgol_filter


from Distributions  import Lorentzian, Gaussian
from Histogram      import GetHistogramData ,GetCenters, GetXrange, PlotHistogram
from Fitting        import FitHistogram
from Wrangling      import Filter ,Cutoff ,AllowedQuantumNumbers ,PlotLE
from Estimates      import MeasureFWHM ,PlotGamma ,GammaEstimate
from Operations     import SavGol, SplineSmoothing

def LifeTime(df,J,v,ef,NSigma,bins):
    '''
    Returns lifetime for a given energy level with Cutoff, NSigma

    Inputs:
        df          = Full data set             : pd.DataFrame 
        J           = J quantum number          : value (float) 
        v           = v quantum number          : value (int) 
        ef          = e/f quantum number        : value (str)
        NSigma      = Number of Std for cutoff  : value (float)
        bins        = Number of Bins            : value (float)   
    Outputs:
        Lifetime    = Lifetime                  : value (float)
        Gamma_Guess = Gamma estimate            : value (float)
        Gamma_Fit   = Gamma from fit            : value (float)
    '''
    L,E = Filter(df, J, v, ef)

    L,E = Cutoff(L,E,NSigma=NSigma)

    Count, Edges, Mean = GetHistogramData(E, bins)

    Gamma_Guess = GammaEstimate(Edges, Count, Plot = False)
    x0_Guess    = 1

    guesses = [Gamma_Guess,x0_Guess]
    
    popt, Lifetime = FitHistogram(Count, Edges,guesses = guesses)
    
    Gamma_Fit = popt[1]

    return Lifetime, Gamma_Guess, Gamma_Fit

def LifeTimeOverBins(df, J, v, ef, NSigma, Bins, progress_bar = False):
    '''
    Calculates the lifetime as a function of the number of bins in the histogram for a given cutoff NSigma. 

    As some fits fail, that bin number is removed from the ActiveBins object

    Inputs:
        df          = Full data set             : pd.DataFrame 
        J           = J quantum number          : value (float) 
        v           = v quantum number          : value (int) 
        ef          = e/f quantum number        : value (str)        
        NSigma      = Number of Std for cutoff  : value (float)
        Bins        = Range of bin number       : list  (float)   
    
    Outputs:
        ActiveBins  = Bin numbers for which a fit exists : list (int)
        Lifeitmes   = Lifetimes by bin number
    '''
    Lifetimes = [] 
    ActiveBins = []
    
    if progress_bar == True:
        Bins = tqdm(Bins, desc=f"NSigma = {NSigma}, J = {J}, v = {v}, e/f = {ef}")

    for bin in Bins:
        try:
            Lifetimes.append(LifeTime(df,J,v,ef,NSigma,bin)[0])
            ActiveBins.append(bin)
        except:
            pass
    return ActiveBins, Lifetimes

def StatisticalLifetime(Lifetimes, Bins, params=False):
    '''
    Uses the lifetime over the bins to produce a statistical value for the lifetime with an uncertainty by fitting a Gaussian to the spread of lifetimes. 
    Inputs:
        Lifeitmes           = Lifetimes by bin number        : list  (float)
        Bins                = Range of bin number            : list  (float)   
    Outputs:
        MeasuredLifetime    = Mean of Gaussian               : value (float)   
        Uncertainty         = Standard deviation of Gaussian : value (float)   
    '''

    if len(Lifetimes) < len(Bins)/3:
        MeasuredLifeitme = np.nan
        Uncertainty      = np.nan
        return MeasuredLifeitme, Uncertainty
        #raise ValueError(f"More than half your fits failed. Out of {len(Bins)}, {len(Lifetimes)} Succeeded")
    
    bin_number = int(np.floor(len(Lifetimes)/10))

    Count, Edges, Mean = GetHistogramData(np.array(Lifetimes),bin_number,Centered=False)

    Centers = GetCenters(Edges)

    EstimatedMean = np.mean(Lifetimes)
    EstimatedSTD  = np.std(Lifetimes)

    popt = curve_fit(Gaussian,Centers, Count, p0 = [EstimatedMean, EstimatedSTD])[0]

    MeasuredLifeitme = popt[0]  
    Uncertainty      = popt[1]  

    return (MeasuredLifeitme, Uncertainty) if params ==False else (MeasuredLifeitme, Uncertainty, popt, Count, Edges, Centers)


# %%