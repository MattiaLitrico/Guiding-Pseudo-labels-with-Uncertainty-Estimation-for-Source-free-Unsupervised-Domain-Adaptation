from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np
import torchvision.transforms as transforms
import wandb
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import *
from utils import *
from torchvision.models import resnet18, resnet50, resnet101
from datasets import *
from model import *
from moco import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_neighbors', default=10, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--temporal_length', default=5, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--source', default='cifar10', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--noisy_path', type=str, default=None)
parser.add_argument('--run_name', type=str)
parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--wandb', action='store_true', help="Use wandb")

parser.add_argument('--ctr', action='store_false', help="use contrastive loss")
parser.add_argument('--label_refinement', action='store_false', help="Use label refinement")
parser.add_argument('--neg_l', action='store_false', help="Use negative learning")
parser.add_argument('--reweighting', action='store_false', help="Use reweighting")

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.wandb:
    wandb.init(project="Guiding Pseudo-labels with Uncertainty Estimation for Test-Time Adaptation", name = args.run_name)

def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim=axis)

def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank):
    pred_probs = []
    pred_probs_all = []

    for feats in features.split(64):
        distances = get_distances(feats, features_bank)
        _, idxs = distances.sort()
        idxs = idxs[:, : args.num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)
    
    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    #First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard

def refine_predictions(
    features,
    probs,
    banks):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(
        features, feature_bank, probs_bank
    )

    return pred_labels, probs, pred_labels_all, pred_labels_hard

def contrastive_loss(logits_ins, pseudo_labels, mem_labels):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2) 
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    return loss

@torch.no_grad()
def update_labels(banks, idxs, features, logits):
    probs = F.softmax(logits, dim=1)

    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])

def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div

def nl_criterion(output, y):
    output = torch.log( torch.clamp(1.-F.softmax(output, dim=1), min=1e-5, max=1.) )
    
    labels_neg = ( (y.unsqueeze(-1).repeat(1, 1) + torch.LongTensor(len(y), 1).random_(1, args.num_class).cuda()) % args.num_class ).view(-1)

    l = F.nll_loss(output, labels_neg, reduction='none')

    return l

# Training
def train(epoch, net, moco_model, optimizer, trainloader, banks):
    loss = 0
    acc = 0

    net.train()
    moco_model.train()

    for batch_idx, batch in enumerate(trainloader): 
        weak_x = batch[0].cuda()
        strong_x = batch[1].cuda()
        y = batch[2].cuda()
        idxs = batch[3].cuda()
        strong_x2 = batch[5].cuda()

        feats_w, logits_w = moco_model(weak_x, cls_only=True)

        if args.label_refinement:
            with torch.no_grad():
                probs_w = F.softmax(logits_w, dim=1)
                pseudo_labels_w, probs_w, _, _ = refine_predictions(feats_w, probs_w, banks)
        else:
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w = probs_w.max(1)[1]
        
        _, logits_q, logits_ctr, keys = moco_model(strong_x, strong_x2)

        if args.ctr:
            loss_ctr = contrastive_loss(
                logits_ins=logits_ctr,
                pseudo_labels=moco_model.mem_labels[idxs],
                mem_labels=moco_model.mem_labels[moco_model.idxs]
            )
        else:
            loss_ctr = 0
        
        # update key features and corresponding pseudo labels
        moco_model.update_memory(epoch, idxs, keys, pseudo_labels_w, y)

        with torch.no_grad():
            #CE weights
            max_entropy = torch.log2(torch.tensor(args.num_class))
            w = entropy(probs_w)

            w = w / max_entropy
            w = torch.exp(-w)
        
        #Standard positive learning
        if args.neg_l:
            #Standard negative learning
            loss_cls = ( nl_criterion(logits_q, pseudo_labels_w)).mean()
            if args.reweighting:
                loss_cls = (w * nl_criterion(logits_q, pseudo_labels_w)).mean()
        else:
            loss_cls = ( CE(logits_q, pseudo_labels_w)).mean()
            if args.reweighting:
                loss_cls = (w * CE(logits_q, pseudo_labels_w)).mean()

        loss_div = div(logits_w) + div(logits_q)

        l = loss_cls + loss_ctr + loss_div

        update_labels(banks, idxs, feats_w, logits_w)

        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        accuracy = 100.*accuracy_score(y.to('cpu'), logits_w.to('cpu').max(1)[1])

        loss += l.item()
        acc += accuracy
        
        if batch_idx % 100 == 0:
            print('Epoch [%3d/%3d] Iter[%3d/%3d]\t ' 
                    %(epoch, args.num_epochs, batch_idx+1, len(trainloader)))

            print("Acc ", acc/(batch_idx+1))

    
    print("Training acc = ", acc/len(trainloader))

    if args.wandb:
        wandb.log({
        'train_loss': loss_cls/len(trainloader), \
        'train_acc': acc/len(trainloader), \
        }, step=epoch) 

@torch.no_grad()
def eval_and_label_dataset(epoch, model, banks):
    model.eval()
    logits, indices, gt_labels = [], [], []
    features = []

    for batch_idx, batch in enumerate(test_loader):
        inputs, targets, idxs = batch[0].cuda(), batch[2].cuda(), batch[3].cuda()
        
        feats, logits_cls = model(inputs, cls_only=True)

        features.append(feats)
        gt_labels.append(targets)
        logits.append(logits_cls)
        indices.append(idxs)            

    features = torch.cat(features)
    gt_labels = torch.cat(gt_labels)
    logits = torch.cat(logits)
    indices = torch.cat(indices)

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: 16384],
        "probs": probs[rand_idxs][: 16384],
        "ptr": 0,
    }

    # refine predicted labels
    pred_labels, _, _, _ = refine_predictions(features, probs, banks) 

    acc = 100.*accuracy_score(gt_labels.to('cpu'), pred_labels.to('cpu'))          

    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  

    if args.wandb:
        wandb.log({
        'val_accuracy': acc, \
        }, step=epoch)
    
    return acc, banks, gt_labels, pred_labels

def create_model(arch, args):
    model = Resnet(arch, args)

    model = model.cuda()
    return model

arch = 'resnet18'

if args.dataset.split('/')[0] == 'pacs':
    train_dataset = dataset(dataset=args.dataset, root='./data/PACS', noisy_path=None,
                          mode='all',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset=args.dataset, root='./data/PACS', noisy_path=None,
                         mode='all',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset.split('/')[0] == 'visdac':
    train_dataset = dataset(dataset=args.dataset, root='./data/VisdaC', noisy_path=None,
                          mode='train',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset=args.dataset, root='./data/VisdaC', noisy_path=None,
                         mode='test',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )
    
    arch = 'resnet101'

elif args.dataset.split('/')[0] == 'domainnet':
    train_dataset = dataset(dataset=args.dataset, root='./data/domainnet', noisy_path=None,
                          mode='all',
                          transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )
    
    test_dataset = dataset(dataset=args.dataset, root='./data/domainnet', noisy_path=None,
                         mode='all',
                         transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )
    
    arch = 'resnet50'

logdir = 'logs/' + args.run_name
net = create_model(arch, args)
momentum_net = create_model(arch, args)

load_weights(net, 'logs/' + args.source + '/weights_best.tar')
load_weights(momentum_net, 'logs/' + args.source + '/weights_best.tar')

optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4)

moco_model = AdaMoCo(src_model = net, momentum_model = momentum_net, features_length=net.bottleneck_dim, num_classes=args.num_class, dataset_length=len(train_dataset), temporal_length=args.temporal_length)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cudnn.benchmark = True

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=4,
                                               drop_last=True,
                                               shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=4,
                                              drop_last=True,
                                              shuffle=False)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()

best = 0

acc, banks, _, _ = eval_and_label_dataset(0, moco_model, None)

for epoch in range(args.num_epochs+1):
 
    print('Train Nets')
    train(epoch, net, moco_model, optimizer, train_loader, banks) # train net1  

    acc, banks, gt_labels, pred_labels = eval_and_label_dataset(epoch, moco_model, banks)

    if acc > best:
        save_weights(net, epoch, logdir + '/weights_best.tar')
        best = acc
        print("Saving best!")

        if args.wandb:
            wandb.run.summary['best_acc'] = best


