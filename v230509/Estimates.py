# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from Histogram import GetCenters

def MeasureFWHM(Edges, Count, NumPoints=None):
    '''
    Measures the FWHM by interpolating Lorentzian distribution and finding intersection points.

    Inputs:
        Edges     = Bin edges of histogram pdf : np.1darray (float)
        Count     = Counts for histogram pdf   : np.1darray (float)
        NumPoints = Number of bins             : value      (int)
    Outputs: 
        Erange    = Interpolation x values     : np.1darray (float)
        Hrange    = Interpolation y values     : np.1darray (float)
        min_idx   = Index of frst intersection : value      (int)
        max_idx   = Index of scnd intersection : value      (int)
        FWHM_Interp_Measurement = MeasuredFWHM : value      (float)
    '''

    Centers = GetCenters(Edges)
    
    NumPoints = 10000 if NumPoints is None else NumPoints

    Erange = np.linspace(min(Centers), max(Centers), NumPoints)

    H = interpolate.interp1d(Centers,Count)
    Hrange = H(Erange)

    InterpFWHM  = max(Hrange)/2 * np.ones(NumPoints)

    idx_Interp  = np.argwhere(np.diff(np.sign(InterpFWHM - Hrange))).flatten()

    if len(idx_Interp) > 2:
       raise ValueError("There are more than two intersection points, try lowering your bins, check your histogram.")
    
    min_idx = idx_Interp[0]
    max_idx = idx_Interp[1]
    FWHM_Interp_Measurement = abs(Erange[min_idx]-Erange[max_idx])

    return Erange, Hrange, min_idx, max_idx, FWHM_Interp_Measurement

def PlotGamma(Erange, Hrange, min_idx, max_idx):
    '''
    Plots FWHM of Lorentzian distribution as shaded area. 

    Inputs: 
        Erange    = Interpolation x values     : np.1darray (float)
        Hrange    = Interpolation y values     : np.1darray (float)
        min_idx   = Index of frst intersection : value      (int)
        max_idx   = Index of scnd intersection : value      (int)
    Oututs:
        None
    '''
    plt.figure(figsize=(16,9))

    plt.title("Measurement of FWHM for Lorentzian Distribution",fontsize = 30)

    plt.plot(Erange, Hrange, "k--", label = "Interpolated Distribution")
    #plt.plot(Erange[min_idx:max_idx+1], InterpFWHM[min_idx:max_idx+1], "b-", label = "FWHM")

    plt.xlabel(r"$E-\langle E \rangle$ / cm $^{-1}$",fontsize = 30)
    plt.ylabel("Count",fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)

    #plt.legend(loc = "best", fontsize = 20)
    
    plt.axhline(min(Hrange), color="k" ,alpha = 0.5)
    plt.fill_between(Erange, Hrange, min(Hrange),
            where = (Erange >= Erange[min_idx]) & (Erange <= Erange[max_idx]),
            color = 'g')

def GammaEstimate(Edges, Count, **kwargs):
    '''
    Wrapper function to extract FWHM estimate and Plotting

    Inputs: 
        Edges     = Bin edges of histogram pdf : np.1darray (float)
        Count     = Counts for histogram pdf   : np.1darray (float)
    Outputs:
        FWHM_Interp_Measurement = MeasuredFWHM : value      (float)
    '''
    if "NumPoints" in kwargs.keys():
        NumPoints = kwargs["NumPoints"]
    else:
        NumPoints = None

    Erange, Hrange, min_idx, max_idx, FWHM_Interp_Measurement = MeasureFWHM(Edges, Count, NumPoints)
    
    if "Plot" in kwargs.keys():
        Plot = kwargs["Plot"]
    else:
        Plot = False

    if Plot == True:
        PlotGamma(Erange, Hrange, min_idx, max_idx)
    
    return FWHM_Interp_Measurement


    
