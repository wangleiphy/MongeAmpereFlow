import math 
import torch
torch.manual_seed(42)
import io 
import torch.nn.functional as F
from torch.autograd import grad as torchgrad
from torchvision.utils import make_grid, save_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from flow import MongeAmpereFlow
from objectives import Ising, Phi4
from net import MLP, CNN, Simple_MLP 
from utils import save_checkpoint, load_checkpoint
from symmetrize import Symmetrize
from paper import inverserg, sample

def vi(target, model, optimizer, Nepochs, Batchsize, L, alpha = 0.0 , delta = 0.0, 
       save = True, save_period=10, 
       fe_exact = None, obs_exact = None, device = 'cpu'
       ):

    LOSS = []
    
    params = list(model.parameters()) 
    #filter out those we do not want to train
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
   
    plt.ion() 

    #samples 
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    l1, = ax1.plot([], [],'o', alpha=0.5, label='direct generated')
    #l2, = ax1.plot([], [],'s', alpha=0.5, label='latent space hmc')
    #l21, = ax1.plot([], [],'*', alpha=0.5, label='physical space hmc')

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.legend(loc='upper left')
    fig1.canvas.draw()

    #loss
    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(311)
    l3, = ax2.plot([], [], label='fe std')
    ax2.set_xlim([0, Nepochs])
    ax2.legend()

    ax3 = fig2.add_subplot(312)
    l4, = ax3.plot([], [], label='fe mean')
    l41, = ax3.plot([], [], label='loss')
    if (fe_exact is not None):
        plt.axhline(fe_exact, color='r', lw=2)
    ax3.set_xlim([0, Nepochs])
    ax3.legend()

    ax31 = fig2.add_subplot(313)
    l5, = ax31.plot([], [], label='force diff')
    ax31.set_xlim([0, Nepochs])
    ax31.legend()
    fig2.canvas.draw()

    #contour
    #build up mesh
    xlimits=[-5, 5]
    ylimits=[-5, 5]
    numticks=31
    x = np.linspace(*xlimits, num=numticks, dtype=np.float32)
    y = np.linspace(*ylimits, num=numticks, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    xy = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
    z = torch.zeros(xy.shape[0], L**2, device=device)
    z[:, 0]= torch.from_numpy(xy[:, 0])
    z[:, 1]= torch.from_numpy(xy[:, 1])

    #fig3 = plt.figure()
    #ax4 = fig3.add_subplot(111)
    #plt.xlabel('$x_1$')
    #plt.ylabel('$x_2$')
    #plt.xlim(xlimits)
    #plt.ylim(ylimits)
    #plt.legend(loc='upper left')
    #fig3.canvas.draw()

    fig4 = plt.figure()
    ax5 = fig4.add_subplot(111)
    im = ax5.imshow(np.zeros((1, 1)), cmap=cm.gray)
    
    with io.open(model.name + '.log', 'a', buffering=1, newline='\n') as logfile:
        for epoch in range(Nepochs):
            
            x, logp_x = model.sample(Batchsize) # sample from the model
            logpi_x = target(x)                 # target 
            fe = (logp_x - logpi_x)/L**2        # actual free energy
            fe_anneal = (logp_x - logpi_x)/L**2 # annealed free energy
            
            if (delta>0):
                #compute force difference
                force_model = torch.autograd.grad(-model.nll(x), x, grad_outputs=torch.ones(x.shape[0], device=x.device))[0]
                force_target = torch.autograd.grad(target(x), x, grad_outputs=torch.ones(x.shape[0], device=x.device))[0]
                force_diff = ((force_model-force_target)**2).mean()
                loss = fe_anneal.mean() + delta * force_diff

            else:
                force_diff = torch.Tensor([np.NaN]) 
                loss = fe_anneal.mean() 

            message = 'epoch: {}, loss: {:.6f}, fe: {:.6f}, fe_std: {:.6f}, force_diff: {:.6f}'.format(epoch,
                                                                                                       loss.data.item(), 
                                                                                                       fe.mean().data.item(), 
                                                                                                       fe.std().data.item(), 
                                                                                                       force_diff.data.item())

            print (message)

            message = ('{} '+ 4*'{:.6f} ').format(epoch,
                                                loss.data.item(), 
                                                fe.mean().data.item(), 
                                                fe.std().data.item(), 
                                                force_diff.data.item())
            logfile.write(message + u'\n')

            LOSS.append([fe.std().data.item(), 
                         fe.mean().data.item(),
                         loss.data.item(),
                         force_diff.data.item()
                         ])
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if save and epoch%save_period==0:
                #with torch.no_grad():
                if True: 
                    save_checkpoint(model.name+'/epoch_{:04d}.chkp'.format(epoch), model, optimizer)
                    
                    #fig1: sample projection
                    proposals = x.data.cpu().numpy()
                    proposals.shape = (Batchsize, -1)
                    
                    l1.set_xdata(proposals[:,0])
                    l1.set_ydata(proposals[:,1])
                    ax1.set_title('epoch=%g'%(epoch))
                    ax1.relim()
                    ax1.autoscale_view() 
                    fig1.canvas.draw()
                    fig1.savefig(model.name+'/projection_{:04d}.png'.format(epoch)) 
                    
                    #fig2: losses
                    loss4plot = np.array(LOSS).reshape(-1, 4)
                    l3.set_xdata(range(len(LOSS)))
                    l3.set_ydata(loss4plot[:, 0])
                    l4.set_xdata(range(len(LOSS)))
                    l4.set_ydata(loss4plot[:, 1])
                    l41.set_xdata(range(len(LOSS)))
                    l41.set_ydata(loss4plot[:,2])
                    l5.set_xdata(range(len(LOSS)))
                    l5.set_ydata(loss4plot[:, 3])

                    plt.xlabel('epochs')

                    ax2.relim()
                    ax2.autoscale_view() 

                    ax3.relim()
                    ax3.autoscale_view() 

                    ax31.relim()
                    ax31.autoscale_view() 
                    fig2.canvas.draw()
                    fig2.savefig(model.name + '/loss.png')
                
                    #fig3: contour 
                    #ax4.cla()
                    #zs = model(z).data.cpu().numpy()
                    #Z = zs.reshape(X.shape)
                    #ax4.contour(X, Y, Z, alpha=0.8)
                    #fig3.canvas.draw()

                    #fig4: configurations
                    x, _ = model.sample(100) # samples 
                    p = x.view(-1, 1, L, L) 
                    #p = torch.sigmoid(2.*x).view(x.shape[0], 1, L, L) # put it into 0-1
                    img = make_grid(p, padding=1, nrow=10,normalize=True,scale_each=False).to('cpu').detach().numpy()
                    im.set_data(np.transpose(img, (1, 2, 0)))
                    fig4.canvas.draw()
                    save_image(p, model.name+'/config_{:04d}.png'.format(epoch), nrow=10, padding=1)
                    plt.pause(0.001)

        return model

if __name__=="__main__":
    import h5py
    import subprocess
    import argparse
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-folder", default='data/',
                    help="where to store results")

    group = parser.add_argument_group('learning  parameters')
    group.add_argument("-Nepochs", type=int, default=2000, help="")
    group.add_argument("-Batchsize", type=int, default=256, help="")
    group.add_argument("-cuda", type=int, default=-1, help="use GPU")
    group.add_argument("-double", action='store_true', help="use float64")
    group.add_argument("-lr", type=float, default=0.001, help="learning rate")
    group.add_argument("-decay", type=float, default=0.001, help='weight decay rate')
    group.add_argument("-alpha", type=float, default=0.0, help="reg term")
    group.add_argument("-delta", type=float, default=0.0, help="reg term")
    group.add_argument("-save_period", type=int, default=100, help="")

    group = parser.add_argument_group('network parameters')
    group.add_argument("-net", default='Simple_MLP', help="network")
    group.add_argument("-checkpoint", default=None, help="checkpoint")
    #group.add_argument("-prior", default='Gaussian', help="prior distribution")
    #group.add_argument("-nlayers", type=int, default=8, help="# of layers in RNVP block")
    #group.add_argument("-depth", type=int, default=-1, help="-1 means learn mera")
    group.add_argument("-hdim", type=int, default=512, help="")
    group.add_argument("-symmetrize", action='store_true', help="randomly symmetrized spatial symm")

    group = parser.add_argument_group('flow parameters')
    group.add_argument("-Nsteps", type=int, default=10, help="# of integration")
    group.add_argument("-epsilon", type=float, default=0.1, help="integration step")

    group = parser.add_argument_group('target parameters')
    group.add_argument("-target", default='ising', help="target distribution")
    group.add_argument("-L",type=int, default=4,help="linear size")
    group.add_argument("-d",type=int, default=2,help="dimension")
    group.add_argument("-BC", default='periodic', help="boundary condition")

    #ising
    group.add_argument("-T",type=float, default=2.269185314213022, help="Temperature")

    #phi4 
    group.add_argument("-n",type=int, default=1,help="component")
    group.add_argument("-kappa",type=float, default=0.15, help="kappa")
    group.add_argument("-lambd",type=float, default=0.0, help="lambd")

    group.add_argument("-fe_exact",type=float,default=None,help="fe_exact")
    group.add_argument("-obs_exact",type=float,default=None,help="obs_exact")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-train", action='store_true', help="if we train the model")
    group.add_argument("-show", action='store_true',  help="show figure right now")
    group.add_argument("-outname", default="result.pdf",  help="output pdf file")

    args = parser.parse_args()
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))
    print ('device:', device)
    if args.double:
        print ('use float64')
    else:
        print ('use float32')

    if args.target == 'phi4':
        target = Phi4(args.n, args.L, args.d, args.kappa, args.lambd, device=device)
    elif args.target == 'ising':
        target = Ising(args.L, args.d, args.T, args.BC)
    else:
        print ('what target ?', args.target)
        sys.exit(1)

    target.to(device)

    folder = args.folder+ '/learn_ot/'
    key = folder + args.target 

    if (args.target=='ising'):
        key += '_' +args.BC \
              + '_L' + str(args.L)\
              + '_d' + str(args.d) \
              + '_T' + str(args.T)
    elif (args.target=='phi4'):
        key += '_L' + str(args.L)\
              + '_d' + str(args.d) \
              + '_n' + str(args.n) \
              + '_kappa' + str(args.kappa) \
              + '_lambd' + str(args.lambd) 

    if (args.symmetrize):
        key += '_symmetrize'

    key+= '_'+args.net \
          + '_hdim' + str(args.hdim) \
          + '_Batchsize' + str(args.Batchsize) \
          + '_lr' + str(args.lr) \
          + '_delta' + str(args.delta) \
          + '_Nsteps' + str(args.Nsteps) \
          + '_epsilon' + str(args.epsilon) 

    cmd = ['mkdir', '-p', key]
    subprocess.check_call(cmd)
    
    if args.net == 'MLP':
        net = MLP(dim=target.nvars, hidden_size = args.hdim)
    elif args.net == 'CNN':
        net = CNN(L=args.L, hidden_size = args.hdim)
    elif args.net == 'Simple_MLP':
        net = Simple_MLP(dim=target.nvars, hidden_size = args.hdim)
    else:
        print ('what network ?', args.net)
        sys.exit(1)

    if args.symmetrize:
        print ('symmetrized net')
        net = Symmetrize(net, args.L, args.n)
    
    model = MongeAmpereFlow(net, args.epsilon, args.Nsteps, device=device, name = key)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.checkpoint is not None:
        try:
            load_checkpoint(args.checkpoint, model, optimizer)
            print('load checkpoint', args.checkpoint)
        except FileNotFoundError:
            print('checkpoint not found:', args.checkpoint)
    
    if args.train:
        model = vi(target, model, optimizer, args.Nepochs, args.Batchsize, args.L, alpha=args.alpha, delta=args.delta, save_period=args.save_period, fe_exact=args.fe_exact, obs_exact=args.obs_exact, device=device)
        print('#trained model', model.name)

    else:
        inverserg(model)
        #sample(model)

        if args.show:
            plt.show()
        else:
            plt.savefig(args.outname, dpi=300, transparent=True)


