# %%
from statisticallifetimes.Parallelised import Levels, StatLT_AllStates, Store_Levels, Get_Levels
import pandas as pd
import numpy as np

df = pd.read_csv("./Testing.csv")
NSigmas = [3,4]
Bins    = np.arange(100,130,1)

Data = Levels(StatLT_AllStates(df,NSigmas, Bins, head = 5))
# %%
Store_Levels(Data,"Test.slt")
# %%
