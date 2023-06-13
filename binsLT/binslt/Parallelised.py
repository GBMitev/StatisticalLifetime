# %%
from .dependencies import *
from .wrangling import *
from .lifetime import *

def StatLT_Sigma(df, NSigmas, Bins, Quant_Nums):

    J, v, ef = Quant_Nums[0], Quant_Nums[1], Quant_Nums[2]

    SLTs        = {}
    Fit_Infos   = {}

    for NSigma in NSigmas:
        Fit_Info                = lifetime_over_bins(df, J, v, ef, NSigma, Bins)
        Fit_Infos[f"{NSigma}"]  = Fit_Info

        Lifetimes               = Fit_Info["Lifetimes"].to_numpy()
        SLTs[f"{NSigma}"]       = statistical_lifetime(Lifetimes,Bins)
    
    data = [[val[0], val[1]] for val in SLTs.values() if np.isnan(val[0]) == False and np.isnan(val[1])==False]

    if len(data) != 0:
        MeasuredLT, Unc = zip(*data)
        MeasuredLT, Unc = np.array(MeasuredLT), np.array(Unc)

        MeasuredLT_Ave  = np.mean(MeasuredLT)
        Unc_Ave         = 1/len(MeasuredLT)*np.sqrt(sum(Unc**2))
    else:
        MeasuredLT = ufloat(np.nan,np.nan)
        return (J,v,ef, MeasuredLT, SLTs, Fit_Infos)
    
    MeasuredLT = ufloat(MeasuredLT_Ave, Unc_Ave)
    
    return (J, v, ef, MeasuredLT, SLTs, Fit_Infos)

def AQN_Generator(AQN) -> tuple:
    for J, v, ef in AQN.itertuples(index=False):
        yield (J, v, ef)

def StatLT_AllStates(df, NSigmas, Bins, Cores=4, head = None):
    if head is None:
        AQN = allowed_quantum_numbers(df)
    else:
        AQN = allowed_quantum_numbers(df).head(head)
    
    Executor = ProcessPoolExecutor(max_workers=Cores)

    Partial_StatLT_Sigma = partial(StatLT_Sigma,df,NSigmas, Bins)

    with Executor as ex:
        Results = list(tqdm(ex.map(Partial_StatLT_Sigma, AQN_Generator(AQN)), total=len(AQN)))

    return Results
# %%
# CLASSES TO STORE INFORMATION
class Fit_Parameters:
    def __init__(self, dict):
        self.dict = dict

class Levels:
    def __init__(self, Results=None):
        if Results is not None:

            self.Sigmas_Used = [*Results[0][4].keys()]
            Columns = ["J","v","e/f","Lifetime"]+self.Sigmas_Used+["Fit_Info"]

            tuples = []

            for tuple in Results:
                J                   = tuple[0]
                v                   = tuple[1]
                ef                  = tuple[2]
                MeasuredLifetime    = tuple[3]
                SLTs                = tuple[4]
                Fit_Info            = Fit_Parameters(tuple[5])

                SLT_Components = []

                for val in SLTs.values():
                    if val[0] is not None and val[1] is not None:
                        SLT_Components.append(ufloat(val[0],val[1]))
                    else:
                        SLT_Components.append(np.nan)

                tuples.append((J,v,ef,MeasuredLifetime,*SLT_Components,Fit_Info))

            self.FullDataSet = pd.DataFrame(data=tuples, columns=Columns)
        else:
            pass
            
    def LifeTimes(self, **kwargs):
        df = self.FullDataSet[["J","v","e/f","Lifetime",*self.Sigmas_Used,"Fit_Info"]]

        #Progressively filtering
        J = kwargs["J"] if "J" in kwargs.keys() else df["J"]
        df = df[df["J"]==J]

        v = kwargs["v"] if "v" in kwargs.keys() else df["v"]
        df = df[df["v"]==v] 
        
        ef = kwargs["ef"] if "ef" in kwargs.keys() else df["e/f"]
        df = df[df["e/f"]==ef] 
        
        return df
    
    def Full_Fit_Infos(self):
        return self.FullDataSet[["J","v","e/f","Lifetime","Fit_Info"]]
    
    def Get_Fit_Info(self, J, v, ef, NSigma):
        df = self.FullDataSet[["J","v","e/f","Fit_Info"]]
        df = df[
            (df["J"] == J)   & 
            (df["v"] == v)   &
            (df["e/f"] == ef)]
        
        if len(df) > 1:
            print("Warning")
        
        Fit_Info = df["Fit_Info"].to_numpy()[0].dict[f"{NSigma}"]
        return Fit_Info

    def Get_null(self):
        LT = self.LifeTimes()

        null = LT[np.isnan(unumpy.nominal_values(LT["Lifetime"]))==True].reset_index(drop = True)
        return null
    
    def Get_nullnt(self):   
        LT = self.LifeTimes()

        nullnt = LT[np.isnan(unumpy.nominal_values(LT["Lifetime"]))==False].reset_index(drop = True)
        return nullnt
    
    def to_slt(self,fname):
        with open(fname, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_slt(fname):
    with open(fname, 'rb') as handle:
        Lvls = pickle.load(handle)
    Object = Levels()
    Object.__dict__ = Lvls.__dict__
    return Object
# %%
# def UnpackingStatLifeTimes(Levels, NSigmas):
#     Columns = ["J","v","e/f"]+[str(i) for i in NSigmas]

#     SLTs = Levels[["J","v","e/f","StatLifeTimes"]]
#     Levels = Levels[["J","v","e/f","Lifetime","Uncertainty"]]

#     tuples = []
#     for i in SLTs.itertuples(index=False):
#         J   = i[0]
#         v   = i[1]
#         ef  = i[2]
#         SLT = ast.literal_eval(i[3].replace("nan","None"))
#         SLT_Components = []

#         for key, val in SLT.items():
#             if val[0] is not None and val[1] is not None:
#                 SLT_Components.append(ufloat(val[0],val[1]))
#             else:
#                 SLT_Components.append(np.nan)

#         tuples.append((J,v,ef,*SLT_Components))

#     SLTs = pd.DataFrame(data=tuples, columns=Columns)
#     Levels = Levels.merge(SLTs, on = ["J","v","e/f"], how = "inner")

#     return Levels

# def Tau_Gaussian(df, NSigmas, Bins, J, v, ef):
#     StatLifetimes = {}
#     Fit_Infos = {}
#     for NSigma in NSigmas:
#         Fit_Info = LifeTimeOverBins(df, J, v, ef, NSigma, Bins)
#         Fit_Infos    [f"{NSigma}"] = Fit_Info
        
#         Lifetimes = Fit_Info["Lifetimes"].to_numpy()

#         StatLifetimes[f"{NSigma}"] = StatisticalLifetime(Lifetimes, Bins)

#     data = [[val[0], val[1]] for val in StatLifetimes.values() if np.isnan(val[0]) == False and np.isnan(val[1])==False]
#     if len(data) != 0:
#         Tau,Tau_error = zip(*data)

#     else:
#         Tau, Tau_error = np.nan, np.nan
#         return (J,v,ef, Tau,Tau_error,StatLifetimes)

#     Tau = np.array(Tau)
#     Tau_error = np.array(Tau_error)

#     Tau_ave = np.mean(Tau)
#     Tau_ave_error = 1/len(Tau)*np.sqrt(sum(Tau_error**2))

#     Tau = {"Lifetime": Quantity(Tau_ave), "Uncertainty":Quantity(Tau_ave_error)}

#     Tuple = (J,v,ef, Tau_ave,Tau_ave_error,StatLifetimes)
#     return Tuple

# def Tau_Gaussian_Tuple(df, NSigmas, Bins, AQN):
#     return Tau_Gaussian(df, NSigmas, Bins, AQN[0],AQN[1],AQN[2])

# def Tau_Gaussian_MP(df, NSigmas, Bins, Cores=4, head = None):
#     if head is None:
#         AQN = AllowedQuantumNumbers(df)
#     else:
#         AQN = AllowedQuantumNumbers(df).head(head)

#     print("Total Number of States Considered:", len(AQN))

#     Executor = ProcessPoolExecutor

#     Partial_Tau_Gaussian = partial(Tau_Gaussian_Tuple,df,NSigmas, Bins)

#     Results = []

#     with Executor(max_workers=Cores) as ex:
#         for Tuple in ex.map(Partial_Tau_Gaussian,AQN_Generator(AQN)):
#             Results.append(Tuple)

#     Results_df = pd.DataFrame(Results, columns= ["J","v","e/f","Lifetime","Uncertainty","StatLifeTimes"])
    
#     Results_df["StatLifeTimes"] = Results_df["StatLifeTimes"].astype(str)
#     Results_df = UnpackingStatLifeTimes(Results_df, NSigmas)

#     return Results_df

# def Tau_Gaussian_MP_2(df, NSigmas, Bins, Cores=4, head = None):
#     if head is None:
#         AQN = AllowedQuantumNumbers(df)
#     else:
#         AQN = AllowedQuantumNumbers(df).head(head)

#     print("Total Number of States Considered:", len(AQN))
    
#     Executor = ProcessPoolExecutor(max_workers=Cores)

#     Partial_Tau_Gaussian = partial(Tau_Gaussian_Tuple,df,NSigmas, Bins)

#     # Results = []

#     # with Executor(max_workers=Cores) as ex:
#     #     for Tuple in ex.map(Partial_Tau_Gaussian,AQN_Generator(AQN)):
#     #         Results.append(Tuple)

#     with Executor as ex:
#         Results = list(tqdm(ex.map(Partial_Tau_Gaussian, AQN_Generator(AQN)), total=len(AQN)))
    
#     Results_df = pd.DataFrame(Results, columns= ["J","v","e/f","Lifetime","Uncertainty","StatLifeTimes"])
    
#     Results_df["StatLifeTimes"] = Results_df["StatLifeTimes"].astype(str)
#     Results_df = UnpackingStatLifeTimes(Results_df, NSigmas)

#     return Results_df