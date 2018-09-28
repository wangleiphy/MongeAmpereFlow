import torch
import torch.nn as nn 
import numpy as np 

class Symmetrize(nn.Module):
    '''
    argument spatial symmetry (translation and rotation) to the network
    for each input x in the batch,radomly move them to the symmetry realted position via a perm
    f(x) and laplacian f(x) are invariant
    grad f(x) is coinvariant
    '''
    def __init__(self, net, L, n):
        super(Symmetrize, self).__init__()
        self.net = net
        self.L = L 
        self.dim = self.net.dim
        self.n = n 

    def update_perm(self, x):
        #self.sign = 2*torch.randint(2, (x.shape[0], 1), device=x.device) - 1
        offset = np.random.randint(self.L, size=(x.shape[0],2))
        orientation = np.random.randint(2, size=(x.shape[0]))
        #random roll on a 2d lattice as a permutation matrix 
        self.perm = torch.zeros([x.shape[0], self.dim], dtype=torch.int64, device=x.device)
        #parallelize over batch 
        for n in range(self.n):
            for i in range(self.L):
                for j in range(self.L):
                    row = i*self.L+j + n*self.L**2
                    #shift 
                    ii = (i+offset[:, 0])%self.L 
                    jj = (j+offset[:, 1])%self.L
                    #index upto rotation 
                    col = (ii* self.L + jj) *orientation + (jj*self.L +ii)*(1-orientation) + n**self.L**2
                    self.perm[:, row] = torch.from_numpy(col)

        #self.invperm = torch.zeros([x.shape[0], self.dim], dtype=torch.int64, device=x.device)
        #for i in range(self.dim):
        #    self.invperm[range(x.shape[0]), self.perm[:, i]] = i 
        _, self.invperm = torch.sort(self.perm , dim=1) # sort is computationally more complex than inverting by hand, but I trust  low-level code optimization than hand-written loop

    def roll(self, x, direction=1):
        if direction ==1:
            return torch.gather(x, 1, self.perm)
        else:
            return torch.gather(x, 1, self.invperm)

    def forward(self, x):
        y = self.roll(x)
        return self.net.forward(y)

    def grad(self, x):
        '''
        grad u(x)
        '''
        y = self.roll(x)
        out = self.net.grad(y)
        return self.roll(out, -1)

    def laplacian(self, x):
        '''
        div \cdot grad u(x)
        '''
        y = self.roll(x)
        return self.net.laplacian(y)

if __name__=='__main__':
    from net import Simple_MLP
    batchsize = 2
    L = 4
    dim = L**2 
    n = 1
    net = Simple_MLP(dim=dim, hidden_size = 10)
    net = Symmetrize(net, L, n)
    x = torch.rand(batchsize, dim) 
    print (x)
    net.update_perm(x)
    print (net.perm)
    print (net.invperm)
    print (net.roll(x))
