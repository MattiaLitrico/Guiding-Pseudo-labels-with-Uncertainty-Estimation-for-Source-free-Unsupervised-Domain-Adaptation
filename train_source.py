import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
from model import *
import argparse
import numpy as np
import torchvision.transforms as transforms
import wandb
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import *
from utils import *
from torchvision.models import resnet18, resnet50, resnet101
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
    wandb.init(project="Guiding Pseudo-labels with Uncertainty Estimation for Test-Time Adaptation", name = args.run_name)


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

def create_model(arch, args):
    model = Resnet(arch, args)

    model = model.cuda()
    return model

arch = 'resnet18'

if args.dataset.split('/')[0] == 'pacs':
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
net = create_model(arch, args)

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


