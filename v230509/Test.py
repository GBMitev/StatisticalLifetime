from main import *

Level1 = Level(0.5,2,"e",StatLifetimes, Tau)

Total = Levels()
print("Current", Total.df_tot)
Total.AddLevel(Level1)
print("New", Total.df_tot)
# %%
Total.GetLevel(0.5,2,"e")
# %%
L,E = Cutoff(*Filter(df,0.5,4,"e"))
PlotLE(L,E)

Count, Edges, Mean = GetHistogramData(E,100)
PlotHistogram(Edges, Count)
popt, Lifetime = FitHistogram(Count, Edges)
Centers = GetCenters(Edges)

x = np.linspace(min(Centers), max(Centers), 1000)
plt.plot(x, Lorentzian(x, *popt))
# %%
ActiveBins, Lifetimes = LifeTimeOverBins(df, 0.5,2,"e", 2,np.arange(30,300,1))
# %%
Standard_Deviation = np.std(Lifetimes)
data = [(ActiveBins[Lifetimes.index(y)], y) for y in Lifetimes if np.mean(Lifetimes)-Standard_Deviation*3<=y<=np.mean(Lifetimes)+Standard_Deviation*3]
ActiveBins, Lifetimes = zip(*data)

plt.plot(ActiveBins, Lifetimes)
Filtered = SavGol(Lifetimes)
plt.plot(ActiveBins, Filtered)

MLT, ULT = StatisticalLifetime(Lifetimes,Bins=np.arange(30,300,1))
plt.axhline(MLT,color="k")
plt.axhline(MLT+Standard_Deviation)
plt.axhline(MLT-Standard_Deviation)
# %%
L,E = Filter(df,0.5,2,"e")
L,E = Cutoff(L,E,2)
PlotLE(L,E)