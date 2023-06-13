# %%
from .dependencies      import *
from .wrangling         import filter, cutoff, plot_LE
from .histogram         import get_histogram_data, plot_histogram, get_xrange, get_centers
from .distributions     import lorentzian
from .lifetime          import fwhm_to_lifetime, lifetime, lifetime_over_bins, statistical_lifetime
from .utils             import chi_squared, make_pickle, read_pickle

class FitParams:
    def __init__(self, df, sigma):
        self.df     = df
        self.sigma  = sigma
class FitInfos:
    def __init__(self, list_of_fit_params):
        dict = {f"{i.sigma}":i for i in list_of_fit_params}
        
        sigmas = [int(i.sigma) for i in list_of_fit_params]
        
        fit_infos = pd.DataFrame(data = zip(sigmas,list_of_fit_params), columns=["Sigma","FitParams"])
        
        self.fit_infos = fit_infos.set_index("Sigma")

def slt_over_sigmas(df, sigmas, bins, J,v,ef, Duo = None):
    slts        = []
    fit_infos   = []
    for sigma in sigmas:
        fit_info  = lifetime_over_bins(df,J,v,ef,sigma,bins, progress_bar=False)        
        lifetimes = fit_info["Lifetimes"].to_numpy()
        fit_infos.append(FitParams(fit_info,sigma))

        measured_lifetime, uncertainty, _ = statistical_lifetime(lifetimes, bins)
        slts.append(ufloat(measured_lifetime, uncertainty))

    average_slt = ufloat(1,1) #placeholder
    fit_infos = FitInfos(fit_infos)
    return [average_slt, *slts, fit_infos]

def slt_over_levels(df, sigmas, bins, Duo=None, cores = None):
    slt_df = df.groupby(["J","v","e/f","State","Lambda","Sigma"],as_index=False).agg({"E":"mean"})[["J","v","e/f","State","Lambda","Sigma"]].head(16)

    if cores is None:
        pandarallel.initialize(progress_bar=True, verbose = 0)
    else:
        pandarallel.initialize(nb_workers = cores,progress_bar=True, verbose = 0)

    sigma_column_names = [str(i) for i in sigmas]

    slt_df[["Lifetime",*sigma_column_names, "FitInfo"]] =  slt_df.parallel_apply(
        lambda x: slt_over_sigmas(df, sigmas, bins, x["J"],x["v"],x["e/f"])
        ,axis=1
        ,result_type = "expand")
    
    return slt_df

class Sandwich:
    def __init__(self, df, sigmas, bins, cores = None, Duo = None):
        self.df     = df
        self.Duo    = Duo
        self.blt = slt_over_levels(self.df, sigmas, bins, self.Duo, cores = cores)
    
    def to_blt(self,fname):
        with open(fname, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_blt(fname):
    with open(fname, 'rb') as handle:
        Lvls = pickle.load(handle)
    Object = Sandwich()
    Object.__dict__ = Lvls.__dict__
    return Object