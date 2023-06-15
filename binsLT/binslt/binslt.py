# %%
from .dependencies      import *
from .wrangling         import le_wrt_duo
from .lifetime          import lifetime_over_bins, statistical_lifetime


class FitParams:
    def __init__(self, df, sigma):
        self.df     = df
        self.sigma  = sigma
class FitInfos:
    def __init__(self, list_of_fit_params):
        dict = {f"{i.sigma}":i for i in list_of_fit_params}
        
        sigmas = [int(i.sigma) for i in list_of_fit_params]
        
        fit_infos = pd.DataFrame(data = zip(sigmas,list_of_fit_params), columns=["Sigma","FitParams"])
        
        self.fit_infos = fit_infos
    def get_fit_info(self, sigma):
        fit_info = self.fit_infos.loc[self.fit_infos["Sigma"] == sigma]["FitParams"].to_numpy()[0].df
        return fit_info

def slt_over_sigmas(df, sigmas, bins, J,v,ef, Duo = None, lock = None,moct=None):
    slts        = []
    fit_infos   = []
    
    if Duo is not None:
        LE = le_wrt_duo(df, Duo,J,v,ef,lock=lock)
    else:
        LE = None

    for sigma in sigmas:
        fit_info  = lifetime_over_bins(df,J,v,ef,sigma,bins,moct=moct,LE=LE)        
        lifetimes = fit_info["Lifetimes"].to_numpy()
        fit_infos.append(FitParams(fit_info,sigma))

        stat_lifetime, uncertainty, _ = statistical_lifetime(lifetimes, bins)
        slts.append(ufloat(stat_lifetime, uncertainty))

    data = [[val.n,val.s] for val in slts if np.isnan(val.n) == False and np.isnan(val.s)==False]

    if len(data) != 0:
        lifetimes, uncs = zip(*data)
        lifetimes, uncs = np.array(lifetimes), np.array(uncs)

        lifetime_ave    = np.mean(lifetimes)
        unc_ave         = 1/len(lifetimes)*np.sqrt(sum(uncs**2))

        average_slt = ufloat(lifetime_ave,unc_ave) 
    
    if len(data) == 0:
        average_slt = ufloat(np.nan, np.nan)

    fit_infos = FitInfos(fit_infos)

    return [average_slt, *slts, fit_infos]

def slt_over_levels(df, sigmas, bins, Duo=None, cores = None,lock=None,moct=None):
    slt_df = df.groupby(["J","v","e/f","State","Lambda","Sigma"],as_index=False).agg({"E":"mean"})[["J","v","e/f","State","Lambda","Sigma"]]

    if cores is None:
        pandarallel.initialize(progress_bar=True, verbose = 0)
    else:
        pandarallel.initialize(nb_workers = cores,progress_bar=True, verbose = 0)

    sigma_column_names = [str(i) for i in sigmas]

    slt_df[["Lifetime",*sigma_column_names, "FitInfo"]] =  slt_df.parallel_apply(
        lambda x: slt_over_sigmas(df, sigmas, bins
                                  ,x["J"],x["v"],x["e/f"]
                                  ,Duo=Duo,lock=lock,moct=moct)
        ,axis=1
        ,result_type = "expand")
    
    return slt_df

class BinSLT:
    def __init__(self, df, sigmas, bins, cores = None, Duo = None, moct = None, lock = None, **kwargs):
        self.df     = df
        if sigmas is not None:
            self.sigmas_used = [str(i) for i in sigmas]

        if Duo is not None:
            self.Duo = Duo

        if "blt_init" not in kwargs.keys():
            self.blt = slt_over_levels(self.df, sigmas, bins, cores = cores, Duo=Duo, lock=lock, moct=moct)
        elif kwargs["blt_init"] == True:
            pass

    def lifetime(self, J,v,ef):
        df = self.blt[["J","v","e/f","Lifetime",*self.sigmas_used,"FitInfo"]]

        #Progressively filtering
        df = df[(df["J"]==J)&(df["v"]==v)&(df["e/f"]==ef)]
        tau = df["Lifetime"].to_numpy()
        return tau

    def get_fit_info(self,J, v, ef, sigma):
        df = self.blt[["J","v","e/f","FitInfo"]]
        df = df[
            (df["J"] == J)   & 
            (df["v"] == v)   &
            (df["e/f"] == ef)]
        if len(df) > 1:
            print("Warning")
        fit_info = df["FitInfo"].to_numpy()[0].get_fit_info(sigma)
        return fit_info
    
    def get_null(self):
        LT = self.blt[["J","v","e/f","Lifetime"]]
        null = LT[np.isnan(unumpy.nominal_values(LT["Lifetime"]))==True].reset_index(drop = True)

        return null
    
    def get_nullnt(self):   
        LT = self.blt[["J","v","e/f","Lifetime"]]
        nullnt = LT[np.isnan(unumpy.nominal_values(LT["Lifetime"]))==False].reset_index(drop = True)
        
        return nullnt
    
    def to_blt(self,fname):
        with open(fname, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def make_sandwich(self,fname):
        with open(fname, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)        
    
    def get_good_bins(self):
        print("Get them yourself")

def read_blt(fname):
    with open(fname, 'rb') as handle:
        Lvls = pickle.load(handle)
    Object = BinSLT(None, None, None, blt_init=True)
    Object.__dict__ = Lvls.__dict__
    return Object
