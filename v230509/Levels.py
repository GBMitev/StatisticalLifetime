class Level:
    def __init__(self, J,v,ef,StatLifeTimes, Tau):
        self.J              = J
        self.v              = v
        self.ef             = ef
        self.StatLifeTimes  = StatLifeTimes
        self.Tau            = Tau

class Levels:
    def __init__(self):
        self.df = pd.DataFrame(columns=["J","v","e/f","Statistical LifeTime","Uncertainty"])
        self.df_tot = pd.DataFrame(columns=["J","v","e/f","Level"])

    def AddLevel(self, Level):
        J             = Level.J
        v             = Level.v
        ef            = Level.ef
        StatLifetimes = Level.StatLifeTimes
        Tau           = Level.Tau

        StatLifetime  = Tau["Lifetime"]
        Uncertainty   = Tau["Uncertainty"]

        Level_data = {"J":J,"v":v,"e/f":ef,"Statistical LifeTime":StatLifetime,"Uncertainty":Uncertainty}

        Level_df      = pd.DataFrame(Level_data, index = [0])

        self.df       = pd.concat([self.df,Level_df], ignore_index=True)

        Level_tot_data = {"J":J,"v":v,"e/f":ef,"Level":Level}
        Level_df_tot   = pd.DataFrame(Level_tot_data, index = [0])

        self.df_tot   = pd.concat([self.df_tot,Level_df_tot],ignore_index=True)

    def GetLevel(self, J, v, ef):
        Filtered = self.df_tot[(self.df_tot["J"]==J)&(self.df_tot["v"]==v)&(self.df_tot["e/f"]==ef)]
        
        return Filtered["Level"].to_numpy()[0]