# %%
from .Dependencies import *
from .Histogram import GetCenters
from .Distributions import Lorentzian

def FitHistogram(Count,Edges, formatted = True, guesses = None):
    '''
    Fits Lorentzian function to Histrogram Count and Centers and extracts halfwidth

    Inputs: 
        Count       = Count for bin         : np.1darray    (float)
        Edges       = Bin edges             : np.1darray    (float)
        formatted   = format of lifetime    : bool          
    Outputs:
        popt        = fitted parameters     : list          (float)
        lifetime    = lifetime of state     : value         (float or str)
    '''

    Centers = GetCenters(Edges)
    
    if guesses == None:
        guesses = [1,1]
    else:
        guesses = guesses        

    popt = curve_fit(Lorentzian,Centers,Count,p0=guesses)[0]
    HalfWidth = popt[1]

    gamma = HalfWidth*1.98630e-23
    hbar = 1.054571817e-34

    lifetime = hbar/(gamma)

    return [popt, Quantity(lifetime,"s")] if formatted == True else [popt, lifetime]
# %%
