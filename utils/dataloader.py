import torch 
from torchvision import datasets, transforms
from utils import AddUniformNoise, ToTensor, ZeroPadding, Crop, HorizontalFlip, Transpose

def dataloader(dataset, batch_size, cuda):

    if dataset == 'CIFAR10':
        data = datasets.CIFAR10('./CIFAR10', train=True, download=True,
                       transform=transforms.Compose([
                           AddUniformNoise(0.05),
                           Transpose(),
                           ToTensor()
                       ]))

        data_hflip = datasets.CIFAR10('./CIFAR10', train=True, download=True,
                           transform=transforms.Compose([
                           HorizontalFlip(), 
                           AddUniformNoise(0.05),
                           Transpose(),
                           ToTensor()
                       ]))
        data = torch.utils.data.ConcatDataset([data, data_hflip])

        train_data, valid_data = torch.utils.data.random_split(data, [90000, 10000])

        test_data = datasets.CIFAR10('./CIFAR10', train=False, download=True,
                        transform=transforms.Compose([
                            AddUniformNoise(0.05),
                            Transpose(),
                            ToTensor()
                       ]))

    elif dataset == 'MNIST':
        data = datasets.MNIST('./MNIST', train=True, download=True,
                   transform=transforms.Compose([
                       AddUniformNoise(),
                       ToTensor()
                   ]))

        train_data, valid_data = torch.utils.data.random_split(data, [50000, 10000])
 
        test_data = datasets.MNIST('./MNIST', train=False, download=True,
                    transform=transforms.Compose([
                        AddUniformNoise(),
                        ToTensor()
                    ]))
    else:  
        print ('what network ?', args.net)
        sys.exit(1)

    #load data 
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda>-1 else {}

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size, shuffle=True, **kwargs)
 
    test_loader = torch.utils.data.DataLoader(test_data,
        batch_size=batch_size, shuffle=True, **kwargs)
    
    return train_loader, valid_loader, test_loader
