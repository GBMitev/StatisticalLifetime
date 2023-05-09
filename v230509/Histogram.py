# %%
import numpy as np
import matplotlib.pyplot as plt

def GetHistogramData(E, bins, Centered = True):
    '''
    Returns Histogram plot data for distribution of Energy levels of a given state. 

    Inputs: 
        E           = Energies              : np.1darray    (float)    
        bins        = Number of Bins        : value         (float)   
        Centered    = Center on the mean    : bool (default = True)
    
    Outputs:
        Count       = Count for bin         : np.1darray    (float)
        Edges       = Bin edges             : np.1darray    (float)
        Mean        = Dataset mean          : value         (float)
    '''

    Mean = E.mean()    
    E = E-Mean if Centered==True else E

    Count, Edges = np.histogram(E, bins=bins,density=True)

    return Count, Edges, Mean

def GetCenters(Edges):
    '''
    Returns Centers of bins for given Edges

    Inputs: 
        Edges   = Bin edges from np.histogram   : np.1darray (float)
    Outputs:
        Centers = Bin centers from np.histogram : np.1darray (float)
    '''
    Centers = 0.5*(Edges[1:]+ Edges[:-1])
    return Centers 

def PlotHistogram(Edges, Count, Centered = True):
    '''
    Plots Histogram from GetHistogram
    
    Inputs:
        Count       = Count for bin         : np.1darray    (float)
        Edges       = Bin edges             : np.1darray    (float)
        Centered    = Center on mean        : bool
    '''
    plt.figure(figsize = (9,7))
    plt.bar(Edges[:-1], Count,width=np.diff(Edges),color="black", align="edge")
    

    xlabel = r"$E-\langle E \rangle$ / cm $^{-1}$" if Centered == True else r"$E$"
    plt.xlabel(xlabel, fontsize = 20)
    plt.ylabel("Count", fontsize = 20)