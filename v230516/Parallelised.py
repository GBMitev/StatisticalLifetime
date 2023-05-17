# %%
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

def AQN_Generator(AQN) -> tuple:
    for J, v, ef in AQN.itertuples(index=False):
        yield J, v, ef

def UnpackingStatLifeTimes(Levels, NSigmas):
    Columns = ["J","v","e/f"]+[str(i) for i in NSigmas]

    SLTs = Levels[["J","v","e/f","StatLifeTimes"]]
    Levels = Levels[["J","v","e/f","Lifetime","Uncertainty"]]

    tuples = []
    for i in SLTs.itertuples(index=False):
        J   = i[0]
        v   = i[1]
        ef  = i[2]
        SLT = ast.literal_eval(i[3].replace("nan","None"))
        SLT_Components = []

        for key, val in SLT.items():
            if val[0] is not None and val[1] is not None:
                SLT_Components.append(ufloat(val[0],val[1]))
            else:
                SLT_Components.append(np.nan)

        tuples.append((J,v,ef,*SLT_Components))

    SLTs = pd.DataFrame(data=tuples, columns=Columns)
    Levels = Levels.merge(SLTs, on = ["J","v","e/f"], how = "inner")

    return Levels

def Tau_Gaussian_MP(df, NSigmas, Bins, Cores=4, head = None):
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
    
    Results_df["StatLifeTimes"] = Results_df["StatLifeTimes"].astype(str)
    Results_df = UnpackingStatLifeTimes(Results_df, NSigmas)

    return Results_df
# %%