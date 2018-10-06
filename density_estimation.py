import sys 
import io
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from flow import MongeAmpereFlow
from net import MLP, CNN, Simple_MLP 
from utils import save_checkpoint, load_checkpoint
from utils import logit_back, dataloader
from paper import gaussianization, reversibility

np.random.seed(42)

if __name__ == "__main__":
    import h5py
    import subprocess
    import argparse
    from random import randint

    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-folder", default='data/',help="where to store results")
    parser.add_argument("-dataset", default='MNIST', help="")

    group = parser.add_argument_group('learning  parameters')
    group.add_argument("-Nepochs", type=int, default=1000, help="")
    group.add_argument("-Batchsize", type=int, default=100, help="")
    group.add_argument("-cuda", type=int, default=-1, help="use GPU")
    group.add_argument("-double", action='store_true', help="use float64")
    group.add_argument("-lr", type=float, default=0.001, help="learning rate")
    group.add_argument("-decay", type=float, default=0.001, help='weight decay rate')
    group.add_argument("-interactive", action='store_true',  help="show interactive figures")

    group = parser.add_argument_group('network parameters')
    group.add_argument("-net", default='Simple_MLP', help="network")
    group.add_argument("-checkpoint", default=None, help="checkpoint")
    group.add_argument("-hdim", type=int, default=512, help="")

    group = parser.add_argument_group('flow parameters')
    group.add_argument("-Nsteps", type=int, default=10, help="# of integration")
    group.add_argument("-epsilon", type=float, default=0.1, help="integration step")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-train", action='store_true',  help="train")
    group.add_argument("-show", action='store_true',  help="show figure right now")
    group.add_argument("-outname", default="result.pdf",  help="output pdf file")

    args = parser.parse_args()
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))

    if args.double:
        print ('use float64')
    else:
        print ('use float32')
    print ('train on', args.dataset)

    folder = args.folder
    if args.dataset == 'MNIST':
        folder += '/learn_mnist/'
        channel = 1
        length = 28 
        dim = 784
        alpha = 1E-6
    elif args.dataset == 'CIFAR10':
        folder +=  '/learn_cifar10/'
        channel = 3
        length = 32 
        dim = 3072
        alpha = 0.05
    else:
        print ('what dataset ?', args.dataset)
        sys.exit(1)

    key = folder 
    key+= args.net \
          + '_hdim' + str(args.hdim) \
          + '_Batchsize' + str(args.Batchsize) \
          + '_lr' + str(args.lr) \
          + '_Nsteps' + str(args.Nsteps) \
          + '_epsilon' + str(args.epsilon) 

    cmd = ['mkdir', '-p', key]
    subprocess.check_call(cmd)

    if args.net == 'MLP':
        net = MLP(dim=dim, hidden_size = args.hdim, use_z2=False)
    elif args.net == 'CNN':
        net = CNN(L=length, channel=channel, hidden_size = args.hdim, use_z2=False)
    elif args.net == 'Simple_MLP':
        net = Simple_MLP(dim=dim, hidden_size = args.hdim, use_z2=False)
    else:
        print ('what network ?', args.net)
        sys.exit(1)

    model = MongeAmpereFlow(net, args.epsilon, args.Nsteps, device=device, name = key)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    if args.checkpoint is not None:
        try:
            load_checkpoint(args.checkpoint, model, optimizer)
            print('load checkpoint', args.checkpoint)
        except FileNotFoundError:
            print('checkpoint not found:', args.checkpoint)

    params = list(model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)

    train_loader, valid_loader, test_loader = dataloader(args.dataset, args.Batchsize, args.cuda)

    if not args.train:
        model.eval()
        data, _ = list(test_loader)[0]
        data = data.to(device).view(-1, dim)
        #gaussianization(data, model)
        reversibility(data, model)

        if args.show:
            plt.show()
        else:
            plt.savefig(args.outname, dpi=300, transparent=True)
    else:
         
        if args.interactive: 
            plt.ion()
            fig1 = plt.figure()
            ax1 = plt.subplot(111)
            im = ax1.imshow(np.zeros((1, 1)), cmap=cm.gray)
            
            fig2 = plt.figure()
            ax5 = plt.subplot()
            l1, = ax5.plot([],[],label = 'train')
            #l2, = ax5.plot([],[],label = 'validation')
            l3, = ax5.plot([],[], 'r', label = 'test')
            ax5.set_xlim([0,args.Nepochs])
            ax5.legend()
            
        def step(loader, train = True):
            if train: 
                model.train()
            else:
                model.eval()
 
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(loader):
                data = data.to(device).view(-1, dim).requires_grad_()
                #if (train and batch_idx > 10): break
                loss = model.nll(data).mean()
                total_loss += loss.data.item()
 
                if train:
                    print("epoch:",epoch,"iteration:",batch_idx,"loss:",loss.data.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
 
            return total_loss/(batch_idx+1.)
        
        TRAIN_LOSS = []
        VALID_LOSS = []
        TEST_LOSS = []
        with io.open(model.name + '.log', 'a', buffering=1, newline='\n') as logfile:
            for epoch in range(args.Nepochs):
                step(train_loader, train = True)
 
                #with torch.no_grad():
                if True:
                    train_loss = step(train_loader, train= False) # not as the running average 
                    valid_loss = step(valid_loader,train = False)
                    test_loss = step(test_loader,train = False)
                
                    TRAIN_LOSS.append(train_loss)
                    VALID_LOSS.append(valid_loss)
                    TEST_LOSS.append(test_loss)
                    
                    message = ('loss (Training,Validation,Test): '+ 3*'{:.6f}, ').format(train_loss, 
                                                                              valid_loss, 
                                                                              test_loss)

                    print (message)
                    message = ('{} '+ 3*'{:.6f} ').format(epoch, train_loss, 
                                                                 valid_loss,
                                                                 test_loss)
                    logfile.write(message + u'\n')

                    #sample 
                    x, _ = model.sample(16) # samples 
                    x = logit_back(x, alpha).view(x.shape[0], channel, length, length)
                    save_image(x, key+'/epoch_{:04d}.png'.format(epoch), nrow=4, padding=1)
                    save_checkpoint(key+'/epoch_{:04d}.chkp'.format(epoch), model, optimizer)

                    if args.show: 
                        img = make_grid(x, padding=1, nrow=10,normalize=True,scale_each=False).to('cpu').numpy()
                        im.set_data(np.transpose(img, (1, 2, 0)))
                        
                        plt.title('epoch=%g'%epoch)
                        l1.set_xdata(range(len(TRAIN_LOSS)))
                        l1.set_ydata(np.array(TRAIN_LOSS))
                        l2.set_xdata(range(len(VAL_LOSS)))
                        l2.set_ydata(np.array(VAL_LOSS))
                        l3.set_xdata(range(len(TEST_LOSS)))
                        l3.set_ydata(np.array(TEST_LOSS))
                        ax5.relim()
                        ax5.autoscale_view()
                        
                        fig1.canvas.draw()
                        #fig1.savefig(key+'/epoch%g.png'%(epoch)) 

                        fig2.canvas.draw()
                        fig2.savefig(key+'/loss.png') 
                        plt.pause(0.001)

