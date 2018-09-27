import math
import torch 
from utils import logit_back
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np 
import sys 

def sample(model):
    '''
    sample from latent to data 
    '''
    with torch.no_grad():
        x, _ = model.sample(100) # sample with given temperature
        p = torch.sigmoid(2.*x).view(-1, 1, 16, 16)
        s = torch.bernoulli(p)
        img = make_grid(s, padding=1, nrow=10,normalize=True,scale_each=False).to('cpu').numpy()
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.imshow(np.transpose(img, (1, 2, 0)), cmap=cm.gray)
        ax.axis('off')

def inverserg(model):
    '''
    sample from latent to data 
    '''
    with torch.no_grad():
        x = torch.Tensor(4, 256).normal_().requires_grad_().to('cpu')
        logp = -0.5 * x.pow(2).add(math.log(2 * math.pi)).sum(1) 
        imglist = [x.view(-1, 1, 16, 16)]
        for steps in range(5):
            x, logp = model.integrate(x, logp, sign=1, Nsteps=10)
            imglist.append(x.view(-1, 1, 16, 16))
        print (logp)
       
        #imglist = torch.cat(imglist).view(-1, 1, 16, 16)
        #since we plot evolution from left to right
        #need to put batch dimension into 0-th 
        imglist = torch.stack(imglist)
        imglist = imglist.permute(1, 0, 2, 3, 4).contiguous().view(-1, 1, 16, 16)

        img = make_grid(imglist, padding=1, nrow=6, normalize=True, scale_each=False).to('cpu').numpy()
        fig = plt.figure()

        ax1 = plt.subplot(121)
        ax1.imshow(np.transpose(img, (1, 2, 0)), cmap=cm.gray)
        ax1.axis('off')

        #sample spin
        p = torch.sigmoid(2.*x)
        s = torch.bernoulli(p)
        img = make_grid(s.view(-1, 1, 16, 16), padding=1, nrow=1, normalize=True, scale_each=False).to('cpu').numpy()
        ax2 = plt.subplot(122)
        ax2.imshow(np.transpose(img, (1, 2, 0)), cmap=cm.gray)
        ax2.axis('off')

def gaussianization(data, model):
    '''
    inference from data to latent
    '''
    with torch.no_grad():
        print (data.shape)
        x = data.narrow(0, 0, 4)
        imglist = [x.view(-1, 1, 28, 28)]
        logp = torch.zeros(x.shape[0], device=x.device) 
        
        for steps in range(5):
            x, logp = model.integrate(x, logp, sign=-1, Nsteps=20)
            imglist.append(x.view(-1, 1, 28, 28))

        imglist = torch.cat(imglist)

        img = make_grid(imglist, padding=1, nrow=4, normalize=True, scale_each=False).to('cpu').numpy()
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.imshow(np.transpose(img, (1, 2, 0)), cmap=cm.gray)
        ax.axis('off')

def reversibility(x, model):
    with torch.no_grad():
        logp = torch.zeros(x.shape[0], device=x.device) 
        z, logp = model.integrate(x, logp, sign=-1)
        xprime, logp = model.integrate(z, logp, sign=1)
        print ((x-xprime).abs().max())

        img_x = make_grid(logit_back(x).view(-1, 1, 28, 28), padding=1, nrow=10,normalize=True,scale_each=False).to('cpu').numpy()
        img_z = make_grid(z.view(-1, 1, 28, 28), padding=1, nrow=10,normalize=True,scale_each=False).to('cpu').numpy()
        img_xprime = make_grid(logit_back(xprime).view(-1, 1, 28, 28), padding=1, nrow=10,normalize=True,scale_each=False).to('cpu').numpy()
        print (z.shape)
        fig = plt.figure()
        ax1 = plt.subplot(131)
        ax1.imshow(np.transpose(img_x, (1, 2, 0)), cmap=cm.gray)
        ax2 = plt.subplot(132)
        ax2.imshow(np.transpose(img_z, (1, 2, 0)), cmap=cm.gray)
        ax3 = plt.subplot(133)
        ax3.imshow(np.transpose(img_xprime, (1, 2, 0)), cmap=cm.gray)
