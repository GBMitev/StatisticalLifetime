# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Filter(df:pd.DataFrame,J:float,v:int,ef:str):
    '''
    Returns geometries and energies for a given energy level. 

    Inputs:
        df  = Full data set         : pd.DataFrame 
        J   = J quantum number      : value (float) 
        v   = v quantum number      : value (int) 
        ef  = e/f quantum number    : value (str)

    Outputs:
        L   = Geometries            : np.1darray (float)
        E   = Energies              : np.1darray (float)    
    '''

    df = df[(df["J"]==J)&(df["v"]==v)&(df["e/f"]==ef)].sort_values("L")
    L,E = df["L"].to_numpy(),df["E"].to_numpy()
    
    return L,E

def Cutoff(L: list,E: list,NSigma: float=np.inf):
    '''
    Returns Geometries and Energies for Lower < Energy < Upper. 
    Lower = Mean(E)-Std(E)*NSigma 
    Upper = Mean(E)+Std(E)*NSigma 

    Inputs: 
        L       = Geometries            : np.1darray (float)
        E       = Energies              : np.1darray (float)  
        NSigma  = Number of Std         : value      (float)
    
    Outputs:
        L   = Geometries (Adjusted)     : np.1darray (float)
        E   = Energies   (Adjusted)     : np.1darray (float)  
    '''

    Mean = np.mean(E)
    Std  = np.std(E)

    Upper = Mean+Std*NSigma
    Lower = Mean-Std*NSigma

    data = [[L[num], e] for num, e in enumerate(E) if Lower<= e <=Upper]
    data = np.transpose(data)
    return data

def AllowedQuantumNumbers(df:pd.DataFrame):
    '''
    Returns all quantum number subsets in the total DataFrame

    Inputs:
        df  = Full data set             : pd.DataFrame 
    Outputs:
        QN  = Quantum number subsets    : pd.DataFrame
    '''

    QN = df.groupby(["J","v","e/f"], as_index=False).agg({"L":"count"})[["J","v","e/f","L"]]
    
    if len(QN[QN["L"]!=max(QN["L"])]) != 0:
        raise ValueError("Inconsistent quantum number representation over geometries. Check your data.")
    
    QN = QN[["J","v","e/f"]]

    return QN

def PlotLE(L,E,line = False,J = None, v = None, ef = None):
    plt.figure(figsize = (9,7)) 
    
    fmt = "k." if line == False else "k"
    plt.plot(L,E, fmt)

    if v is not None and J is not None and ef is not None:
        title = f"v = {v}, J = {J}, e/f = {ef}"
        plt.title(title)
    
    plt.xlabel(r"Box Length ($\AA$)", fontsize = 20)
    plt.ylabel(r"Energy cm$^{-1}$", fontsize = 20)

    plt.grid(which = "both")
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
