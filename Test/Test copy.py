# %%
from statisticallifetimes.Parallelised import Levels, StatLT_AllStates, read_slt
import pandas as pd
import numpy as np

df = pd.read_csv("./Testing.csv")
All = read_slt("/home/gmitev/Documents/PhD Yr 3/OHModel/PreDissociation/CalculatingLifetimes/AllCombinations/SLTs/ALL.slt")
# %%
#All.to_slt("Testing123.slt")
read_slt("Testing123.slt").Get_nullnt()