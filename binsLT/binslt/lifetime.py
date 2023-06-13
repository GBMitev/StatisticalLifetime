# %%
from .dependencies  import *
from .distributions import gaussian
from .wrangling     import filter, cutoff
from .histogram     import get_histogram_data, gamma_estimate, get_centers
from .fitting       import fit_histogram

def fwhm_to_lifetime(fwhm):

    gamma = fwhm*1.98630e-23
    hbar = 1.054571817e-34

    lifetime = hbar/(gamma)
    return lifetime

def lifetime(df,J,v,ef,sigma,bins, LE=None):
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
    if LE is None:
        L,E = filter(df, J, v, ef)
        L,E = cutoff(L,E,NSigma=sigma)
    else:
        L,E = LE

    count, edges, mean = get_histogram_data(E, bins)

    fwhm_guess  = gamma_estimate(edges, count, Plot = False)
    x0_Guess    = 1

    guesses = [fwhm_guess,x0_Guess]
    
    popt, lifetime = fit_histogram(count, edges,guesses = guesses)
    
    x0     = popt[0]
    fwhm   = popt[1]

    return lifetime, x0, fwhm, mean

def lifetime_over_bins(df, J, v, ef, sigma, bins, LE = None, progress_bar = False, bar_pos=0):
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
        bins = tqdm(bins, desc=f"NSigma = {sigma}, J = {J}, v = {v}, e/f = {ef}", position = bar_pos)

    for bin in bins:
        try:
            Lifetimes .append(lifetime(df,J,v,ef,sigma,bin, LE)[0])
            ActiveBins.append(bin)

            x0        .append(lifetime(df,J,v,ef,sigma,bin, LE)[1])
            FWHM      .append(lifetime(df,J,v,ef,sigma,bin, LE)[2])
            Mean      .append(lifetime(df,J,v,ef,sigma,bin, LE)[3])
        except:
            pass
    
    fit_info = {"ActiveBins": ActiveBins, "Lifetimes":Lifetimes, "x0":x0, "FWHM":FWHM, "Mean":Mean}
    
    fit_info = pd.DataFrame(fit_info)
    return fit_info

def statistical_lifetime(lifetimes, bins, params=False):
    '''
    Uses the lifetime over the bins to produce a statistical value for the lifetime with an uncertainty by fitting a Gaussian to the spread of lifetimes. 
    Inputs:
        lifeitmes           = Lifetimes by bin number        : list  (float)
        bins                = Range of bin number            : list  (float)   
    Outputs:
        MeasuredLifetime    = Mean of Gaussian               : value (float)   
        Uncertainty         = Standard deviation of Gaussian : value (float)   
    '''

    if len(lifetimes) < len(bins)/3:
        measured_lifetime = np.nan
        uncertainty      = np.nan
        return measured_lifetime, uncertainty, len(lifetimes)
    
    bin_number = int(np.floor(len(lifetimes)/10))
    count, edges, _ = get_histogram_data(np.array(lifetimes),bin_number,Centered=False)

    centers = get_centers(edges)

    estimated_mean = np.mean(lifetimes)
    estimated_std  = np.std(lifetimes)

    try:
        popt = curve_fit(gaussian,centers, count, p0 = [estimated_mean, estimated_std])[0]

        measured_lifetime = popt[0]  
        uncertainty      = popt[1]  

        return (measured_lifetime, uncertainty, len(lifetimes)) if params ==False else (measured_lifetime, uncertainty, popt, count, edges, centers)

    except:
        return (np.nan, np.nan, len(lifetimes))