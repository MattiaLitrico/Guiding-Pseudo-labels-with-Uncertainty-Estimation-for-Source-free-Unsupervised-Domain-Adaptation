from __future__ import print_function
from locale import normalize
import warnings
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import *
from datasets import *
from utils import *
from torchvision.models import resnet18, resnet50, resnet101
import pdb
from os.path import join

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize') 
parser.add_argument('--seed', default=1234)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--dataset', default='svhn', type=str)
parser.add_argument('--run_name', type=str)

args = parser.parse_args()
warnings.filterwarnings("ignore")

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)

def test(net):
    net.eval()

    correct = 0
    total = 0
    it = 0
    preds = []
    lab = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs, real_targets, idxs = batch[0].cuda(), batch[2].cuda(), batch[3]
            
            _, outputs = net(inputs)
          
            idxs = idxs.to('cpu').numpy()

            _, predicted = torch.max(outputs, 1)  
                       
            total += real_targets.size(0)
            correct += predicted.eq(real_targets).cpu().sum().item()
            it += 1

            preds.extend(predicted.detach().cpu().numpy())
            lab.extend(real_targets.detach().cpu().numpy())

            print("{}/{}".format(batch_idx, len(test_loader)))

    acc = 100.*correct/total

    print("\n| Test for %s \t Accuracy: %.2f%%\n" %(args.run_name, acc))
    
    return acc

class Resnet(nn.Module):
    def __init__(self, arch):
        super(Resnet, self).__init__()
        bottleneck_dim = 256

        if arch == 'resnet18':
            self.model = resnet18(pretrained=True)
        elif arch == 'resnet101':
            self.model = resnet101(pretrained=True)

        self.model.fc = nn.Linear(self.model.fc.in_features, bottleneck_dim)
        bn = nn.BatchNorm1d(bottleneck_dim)
        self.encoder = nn.Sequential(self.model, bn)

        self.fc = nn.Linear(bottleneck_dim, args.num_class)

        self.fc = nn.utils.weight_norm(self.fc, dim=0)

    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, 1)

        logits = self.fc(features)
     
        return features, logits

def create_model(arch):
    model = Resnet(arch)

    model = model.cuda()
    return model

arch = 'resnet18'

if args.dataset == 'mnist':
    test_dataset = dataset(dataset='mnist', root='./data/', noisy_path=None,
                         mode='all',
                         transform=transforms.Compose([transforms.Resize((36)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset == 'mnistm':
    test_dataset = dataset(dataset='mnistm', root='./data/', noisy_path=None,
                         mode='all',
                         transform=transforms.Compose([transforms.Resize((36)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset == 'svhn':
    test_dataset = dataset(dataset='svhn', root='./data/', noisy_path=None,
                         mode='all',
                         transform=transforms.Compose([transforms.Resize((36)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset == 'usps':
    test_dataset = dataset(dataset='usps', root='./data/', noisy_path=None,
                         mode='all',
                         transform=transforms.Compose([transforms.Resize((36)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset.split('/')[0] == 'pacs':
    test_dataset = dataset(dataset=args.dataset, root='./data/PACS', noisy_path=None,
                         mode='all',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset.split('/')[0] == 'visdac':
    test_dataset = dataset(dataset=args.dataset, root='./data/VisdaC', noisy_path=None,
                         mode='test',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )
    arch = 'resnet101'

net = create_model(arch)

load_weights(net, join('logs', args.run_name, 'weights_best.tar'))

cudnn.benchmark = True

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=8,
                                              drop_last=False,
                                              shuffle=False)


test(net)  


