# %%
import sys
sys.path.append("..")
# %%
from v230516_2.StatisticalLifetime import *
# %%
# Importing DataFrame
fname = "./Testing.csv"
df = pd.read_csv(fname)

# Example Setup parameters
Min_Bin = 30
Max_Bin = 300
Bins = np.arange(Min_Bin,Max_Bin,1)

Sigmas = [1,2]

Cores = int(input("Number of cores:"))

#Results = Tau_Gaussian_MP(df, Sigmas, Bins, Cores = Cores)
#Results.to_csv("Testing_out.csv")
# %%