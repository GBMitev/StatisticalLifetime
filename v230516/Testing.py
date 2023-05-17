# %%
from Lifetime import *

def PlotTuples(Levels,df, number = 5, type = "sample"):
    if type == "sample":
        Levels = Levels[["J","v","e/f"]].sample(number)
    elif type == "head":
        Levels = Levels[["J","v","e/f"]].head(number)
    else:
        raise ValueError(f"type: {type} not supported, choose 'sample' or 'head'")
    
    for J,v,ef in Levels.itertuples(index=False):
        L,E = Filter(df,J,v,ef)
        
        PlotLE(L,E)

        plt.title(f"J={J}, v={v}, e/f={ef}")

df = pd.read_csv("../Energies/Energies_Adj.csv")
Levels = pd.read_csv("./Levels.csv")[["J","v","e/f","Lifetime","Uncertainty"]]

Active_AX = pd.read_csv("../STF_Only_AX/Active_Levels.csv")
Active_AX["Active"] = True

Active_Tot = pd.read_csv("../STF_Only_AX/Active_Levels.csv")
Active_Tot["Active"] = True

def Categorize(Active, Levels):
    Merged = Active.merge(Levels,on = ["J","v","e/f"], how = "outer")

    Good    = Merged[(Merged["Active"] == True)&(Merged["Lifetime"].isnull()==False)]
    Trivial = Merged[(Merged["Active"] != True)&(Merged["Lifetime"].isnull()==True )]

    Bad     = Merged[(Merged["Active"] != True)&(Merged["Lifetime"].isnull()==False)]
    VeryBad = Merged[(Merged["Active"] == True)&(Merged["Lifetime"].isnull()==True )]

    return Good, Trivial, Bad, VeryBad

VeryBad = Categorize(Active_AX, Levels)[-1]
# # %%
# for E,J,v,ef  in VeryBad[["E","J","v","e/f"]].itertuples(index = False):
#     print(E,J,v,ef)

VeryBad.to_csv("../OnlyAX/Analysis/VeryBad.csv", index = False)

# %%
