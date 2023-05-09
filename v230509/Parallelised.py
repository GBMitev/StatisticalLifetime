# %%
from Lifetime import *
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def Tau_Gaussian(df, NSigmas, Bins, J, v, ef):
    StatLifetimes = {}
    
    for NSigma in NSigmas:
        ActiveBins, Lifetimes = LifeTimeOverBins(df, J, v, ef, NSigma, Bins,progress_bar=True)

        StatLifetimes[f"{NSigma}"] = StatisticalLifetime(Lifetimes, Bins)

    data = [[val[0], val[1]] for val in StatLifetimes.values() if np.isnan(val[0]) == False and np.isnan(val[1])==False]
    if len(data) != 0:
        Tau,Tau_error = zip(*data)

    else:
        Tau, Tau_error = np.nan, np.nan
        return (J,v,ef, Tau,Tau_error,StatLifetimes)

    Tau = np.array(Tau)
    Tau_error = np.array(Tau_error)

    Tau_ave = np.mean(Tau)
    Tau_ave_error = 1/len(Tau)*np.sqrt(sum(Tau_error**2))

    Tau = {"Lifetime": Quantity(Tau_ave), "Uncertainty":Quantity(Tau_ave_error)}

    Tuple = (J,v,ef, Tau_ave,Tau_ave_error,StatLifetimes)
    return Tuple

def Tau_Gaussian_Tuple(df, NSigmas, Bins, AQN):
    return Tau_Gaussian(df, NSigmas, Bins, AQN[0],AQN[1],AQN[2])

def AQN_Generator(AQN: pd.DataFrame) -> tuple:
    for J, v, ef in AQN.itertuples(index=False):
        yield J, v, ef

def Tau_Gaussian_MP(df, NSigmas, Bins, Cores=4, full = False, head = None):
    if head is None:
        AQN = AllowedQuantumNumbers(df)
    else:
        AQN = AllowedQuantumNumbers(df).head(head)

    Executor = ProcessPoolExecutor

    Partial_Tau_Gaussian = partial(Tau_Gaussian_Tuple,df,NSigmas, Bins)

    Results = []

    with Executor(max_workers=Cores) as ex:
        for Tuple in ex.map(Partial_Tau_Gaussian,AQN_Generator(AQN)):
            Results.append(Tuple)

    Results_df = pd.DataFrame(Results, columns= ["J","v","e/f","Lifetime","Uncertainty","StatLifeTimes"])

    return Results_df if full == True else Results_df[["J","v","e/f","Lifetime","Uncertainty"]]
# %%