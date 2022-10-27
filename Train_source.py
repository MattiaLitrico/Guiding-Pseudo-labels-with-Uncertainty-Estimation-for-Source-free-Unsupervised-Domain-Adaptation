import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
import torchvision.transforms as transforms
import wandb
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import *
from torchvision.datasets import SVHN, MNIST
from utils import *
from torchvision.models import resnet18, resnet50, resnet101
import pdb
from datasets import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--alfa', default=0.1, type=float)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--run_name', type=str)
parser.add_argument('--wandb', action='store_true', help="Use wandb")

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.wandb:
    wandb.init(project="UDA Diffusion model", name = args.run_name)

def one_hot_encode(labels, num_classes):
    one_hot_vector = torch.zeros((labels.size()[0], num_classes)).cuda()

    for i in range(one_hot_vector.size()[0]):
        one_hot_vector[i, labels[i]] = 1.0

    return one_hot_vector

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Resnet(nn.Module):
    def __init__(self, arch):
        super(Resnet, self).__init__()
        bottleneck_dim = 256

        if arch == 'resnet18':
            self.model = resnet18(pretrained=True)
        elif arch == 'resnet50':
            self.model = resnet50(pretrained=True)
        elif arch == 'resnet101':
            self.model = resnet101(pretrained=True)

        self.model.fc = nn.Linear(self.model.fc.in_features, bottleneck_dim)
        bn = nn.BatchNorm1d(bottleneck_dim)
        self.encoder = nn.Sequential(self.model, bn)

        self.fc = nn.Linear(bottleneck_dim, args.num_class)

        #self.fc = nn.utils.weight_norm(self.fc, dim=0)

    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, 1)

        logits = self.fc(features)
     
        return features, logits

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        
        resnet = self.encoder[0]
        for module in list(resnet.children())[:-1]:
            backbone_params.extend(module.parameters())
        # bottleneck fc + (bn) + classifier fc
        extra_params.extend(resnet.fc.parameters())
        extra_params.extend(self.encoder[1].parameters())
        extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

def adjust_learning_rate(optimizer, progress):
    """
    Decay the learning rate based on epoch or iteration.
    """
    
    decay = 0.5 * (1.0 + math.cos(math.pi * progress / 2))
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr * decay

    return decay

def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1).mean()

    return loss

# Training
def train(epoch, net, optimizer, trainloader):
    loss = []
    acc = []

    net.train()

    for batch_idx, batch in enumerate(trainloader): 
        x = batch[0].cuda()
        y = batch[2].cuda()
            
        _, outputs = net(x)

        l = smoothed_cross_entropy(outputs, y, args.num_class, args.alfa)
        
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        #step = batch_idx + epoch * len(trainloader)
        #adjust_learning_rate(optimizer, step)
        
        accuracy = 100.*accuracy_score(y.to('cpu'), outputs.to('cpu').max(1)[1])

        loss.append(l.item()) 
        acc.append(accuracy)
   
        print('Epoch [%3d/%3d] Iter[%3d/%3d]\t ' 
                %(epoch, args.num_epochs, batch_idx+1, len(trainloader)))

    loss = np.mean( np.array(loss) )
    acc = np.mean( np.array(acc) )

    print("Training acc = ", acc)

    if args.wandb:
        wandb.log({
        'train_loss': loss, \
        'train_acc': acc, \
        }, step=epoch) 

def test(epoch,net):
    net.eval()
    correct = 0

    total = 0
    it = 0
    
    loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs, targets = batch[0].cuda(), batch[2].cuda()
            
            _, outputs = net(inputs)

            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

            loss += CEloss(outputs, targets)
            it += 1

    acc = 100.*correct/total

    loss = loss/it

    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  

    if args.wandb:
        wandb.log({
        'val_loss': loss, \
        'val_net1_accuracy': acc, \
        }, step=epoch)
    
    return acc

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model(arch):
    model = Resnet(arch)

    model = model.cuda()
    return model

arch = 'resnet18'

if args.dataset == 'mnist':
    train_dataset = dataset(dataset='mnist', root='./data/',
                          mode='train',
                          transform=transforms.Compose([transforms.Resize((36)), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset='mnist', root='./data/',
                         mode='test',
                         transform=transforms.Compose([transforms.Resize((36)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset == 'mnistm':
    train_dataset = dataset(dataset='mnistm', root='./data/',
                          mode='train',
                          transform=transforms.Compose([transforms.Resize((36)), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset='mnistm', root='./data/',
                         mode='test',
                         transform=transforms.Compose([transforms.Resize((36)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset == 'svhn':
    train_dataset = dataset(dataset='svhn', root='./data/',
                          mode='train',
                          transform=transforms.Compose([transforms.Resize((36)), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset='svhn', root='./data/',
                         mode='test',
                         transform=transforms.Compose([transforms.Resize((36)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset == 'usps':
    train_dataset = dataset(dataset='usps', root='./data/',
                          mode='train',
                          transform=transforms.Compose([transforms.Resize((36)), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset='usps', root='./data/',
                         mode='test',
                         transform=transforms.Compose([transforms.Resize((36)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset.split('/')[0] == 'pacs':
    train_dataset = dataset(dataset=args.dataset, root='./data/PACS',
                          mode='train',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset=args.dataset, root='./data/PACS',
                         mode='test',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset.split('/')[0] == 'visdac':
    train_dataset = dataset(dataset=args.dataset, root='./data/VisdaC',
                          mode='train',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset=args.dataset, root='./data/VisdaC',
                         mode='test',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

    arch = 'resnet101'

elif args.dataset.split('/')[0] == 'domainnet':
    train_dataset = dataset(dataset=args.dataset, root='./data/domainnet',
                          mode='train',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset=args.dataset, root='./data/domainnet',
                         mode='test',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

    arch = 'resnet50'

logdir = 'logs/' + args.run_name
net = create_model(arch)

cudnn.benchmark = True

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=2,
                                               drop_last=True,
                                               shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=2,
                                              drop_last=True,
                                              shuffle=False)

optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.5, nesterov=False)
"""backbone_params, extra_params = net.get_params()
optimizer = torch.optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": args.lr,
                    "weight_decay": 1e-4,
                    "nesterov": False,
                },
                {
                    "params": extra_params,
                    "lr": args.lr * 10,
                    "weight_decay": 1e-4,
                    "nesterov": False,
                },
            ]
        )"""

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()

best = 0

for epoch in range(args.num_epochs+1):
 
    print('Train Nets')
    train(epoch, net, optimizer, train_loader) # train net1  

    acc = test(epoch,net) 

    if acc > best:
        save_weights(net, epoch, logdir + '/weights_best.tar')
        best = acc
        print("Saving best!")

        if args.wandb:
            wandb.run.summary['best_acc'] = best


