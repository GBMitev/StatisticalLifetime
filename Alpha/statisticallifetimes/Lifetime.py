# %%
from .Dependencies import *
from .Distributions import Gaussian
from .Wrangling import Filter, Cutoff
from .Histogram import GetHistogramData, GammaEstimate, GetCenters
from .Fitting import FitHistogram

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
        Gamma       = Return Gamma guess and fit: bool    
    Outputs:
        Lifetime    = Lifetime                  : value (float)
        Gamma_Guess = Gamma estimate            : value (float)
        Gamma_Fit   = Gamma from fit            : value (float)
    '''
    L,E = Filter(df, J, v, ef)

    L,E = Cutoff(L,E,NSigma=NSigma)

    Count, Edges, Mean = GetHistogramData(E, bins)

    FWHM_Guess  = GammaEstimate(Edges, Count, Plot = False)
    x0_Guess    = 1

    guesses = [FWHM_Guess,x0_Guess]
    
    popt, Lifetime = FitHistogram(Count, Edges,guesses = guesses)
    
    x0     = popt[0]
    FWHM   = popt[1]

    return Lifetime, x0, FWHM, Mean

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
        Lifeitmes   = Lifetimes by bin number            : list (float)
        CenterShift = Lorentzian center                  : list (float)
        LineMean    = Mean value of E                    : list (float)
    '''
    Lifetimes  = [] 
    ActiveBins = []
    
    x0         = []
    FWHM       = []

    Mean       = []

    if progress_bar == True:
        Bins = tqdm(Bins, desc=f"NSigma = {NSigma}, J = {J}, v = {v}, e/f = {ef}")

    for bin in Bins:
        try:
            Lifetimes .append(LifeTime(df,J,v,ef,NSigma,bin)[0])
            ActiveBins.append(bin)

            x0        .append(LifeTime(df,J,v,ef,NSigma,bin)[1])
            FWHM      .append(LifeTime(df,J,v,ef,NSigma,bin)[2])
            Mean      .append(LifeTime(df,J,v,ef,NSigma,bin)[3])
        except:
            pass
    
    dict = {"ActiveBins": ActiveBins, "Lifetimes":Lifetimes, "x0":x0, "FWHM":FWHM, "Mean":Mean}
    
    Fit_Info = pd.DataFrame(dict)
    return Fit_Info

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

    try:
        popt = curve_fit(Gaussian,Centers, Count, p0 = [EstimatedMean, EstimatedSTD])[0]

        MeasuredLifeitme = popt[0]  
        Uncertainty      = popt[1]  

        return (MeasuredLifeitme, Uncertainty) if params ==False else (MeasuredLifeitme, Uncertainty, popt, Count, Edges, Centers)

    except:
        return (np.nan, np.nan)
    