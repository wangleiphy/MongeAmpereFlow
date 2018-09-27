import math 
import torch
import numpy as np


from .template import Target
from .lattice import Hypercube

class Phi4(Target):
    def __init__(self,n,L,d,kappa,lamb, device='cpu'):
        super(Phi4, self).__init__(n*L**d,'Phi4')

        self.channel = n 
        self.lamb = lamb
        self.nsite = L**d

        self.lattice = Hypercube(L, d, 'periodic')
        K = np.eye(self.nsite) -self.lattice.Adj * kappa
        sign, logdet = np.linalg.slogdet(2.*K)
        print (0.5* math.log(2.*math.pi) - 0.5*logdet/L**d)  # lnZ per site at lamb = 0
        self.K = torch.from_numpy(K).float().to(device)

    def energy(self, x): # actually logp

        x = x.view(-1, self.nsite)
        S = -(torch.mm(x,self.K) * x).view(-1, self.channel*self.nsite)
        S = S.sum(dim=1)

        x = x.view(-1, self.channel, self.nsite)
        S = S - self.lamb * (((x**2).sum(dim=1) -1.0)**2 ).sum(dim=1)

        return S

if __name__=='__main__':
    torch.manual_seed(42)
    L = 4
    d = 2
    n = 1
    kappa = 0.15
    lamb = 0.123

    batch_size = 4 
    target = Phi4(n, L, d, kappa, lamb)
    x = torch.randn(batch_size, n, L, L)
    print (x)
    print (target.energy(x))
