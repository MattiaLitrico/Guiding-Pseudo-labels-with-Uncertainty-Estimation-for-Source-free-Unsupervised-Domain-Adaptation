from __future__ import print_function
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
import math
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
    wandb.init(project="UDA Diffusion model", name = args.run_name)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b

def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim=axis)

def strong_aug(img):

    return transforms.Compose(
            [
                transforms.RandomResizedCrop(28, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(32)
            ]
        )(img)

def adjust_learning_rate(optimizer, progress):
    """
    Decay the learning rate based on epoch or iteration.
    """
    
    decay = 0.5 * (1.0 + math.cos(math.pi * progress / 2))
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr * decay

    return decay

def one_hot_encode(labels, num_classes):
    one_hot_vector = torch.zeros((labels.size()[0], num_classes)).cuda()

    for i in range(one_hot_vector.size()[0]):
        one_hot_vector[i, labels[i]] = 1.0

    return one_hot_vector

class Resnet(nn.Module):
    def __init__(self, arch):
        super(Resnet, self).__init__()
        self.bottleneck_dim = 256

        if arch == 'resnet18':
            self.model = resnet18(pretrained=True)
        elif arch == 'resnet101':
            self.model = resnet101(pretrained=True)
        elif arch == 'resnet50':
            self.model = resnet50(pretrained=True)


        self.model.fc = nn.Linear(self.model.fc.in_features, self.bottleneck_dim)
        bn = nn.BatchNorm1d(self.bottleneck_dim)
        self.encoder = nn.Sequential(self.model, bn)

        self.fc = nn.Linear(self.bottleneck_dim, args.num_class)

        self.fc = nn.utils.weight_norm(self.fc, dim=0)

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
    mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2) #torch.all(pseudo_labels.reshape(-1, 1, 1) != mem_labels.unsqueeze(0), dim=2) # #pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
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

"""def sim_weight(ft, mem_ft, y, output):
    with torch.set_grad_enabled(False):
        
        ft = F.normalize(ft, dim=1)
        mem_ft = F.normalize(mem_ft, dim=1)
        
        batch_size = y.size()[0]
        nn = 5

        cosine_similarity = F.cosine_similarity(ft[:,:,None], mem_ft.t()[None,:,:]) * (1-torch.eye(batch_size).cuda())
        nn_cs, nn_idx = torch.topk(cosine_similarity, k=nn, dim=1)
        #nn_cs = nn_cs / nn_cs.sum(1).view(-1,1)
        
        nn_cs = torch.ones(nn_cs.size(), requires_grad=False).cuda()
        nn_cs = nn_cs / nn_cs.sum(1).view(-1,1)

        nn_y = y[nn_idx]
        nn_output = output[nn_idx]
        
        y_new = ( nn_cs.unsqueeze(-1) * nn_y).sum(1)
        output_new = ( nn_cs.unsqueeze(-1) * nn_output).sum(1)
        
        return y_new.max(1)[1], output_new.max(1)[1]

def info_nce_loss(features, labels, classes):
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        classes = (classes.unsqueeze(0) == classes.unsqueeze(1)).float()
        classes = classes.cuda()

        #features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        classes = classes[~mask].view(classes.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        #Excluding same negative pairs
        classes = classes[~labels.bool()].view(similarity_matrix.shape[0], -1)
        negatives[classes.bool()] = 0
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / args.temperature
        return logits, labels"""

def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div

def nl_criterion(output, y):
    output = torch.log( torch.clamp(1.-F.softmax(output, dim=1), min=1e-5, max=1.) )
    
    """#Modified version of negative learning
    if len(y.size()) > 1:
        all_classes = torch.arange(0,10).unsqueeze(0).repeat_interleave(256,dim=0).cuda()
        #Exclude y classes
        mask = torch.all(all_classes.unsqueeze(1) != y.unsqueeze(2), dim=1).cuda()
        probs = torch.ones_like(mask, dtype=int)
        #Put to 0 the probability for excluded values of y
        probs = torch.where(mask == True, probs, torch.zeros_like(mask, dtype=int).cuda())
        probs = probs/probs.sum(1).reshape(-1,1)
        #Select with probability probs
        labels_neg = torch.multinomial(probs, 1).reshape(-1)
    else:"""
    labels_neg = ( (y.unsqueeze(-1).repeat(1, 1) + torch.LongTensor(len(y), 1).random_(1, args.num_class).cuda()) % args.num_class ).view(-1)

    l = F.nll_loss(output, labels_neg, reduction='none')

    return l

"""def generate(imgs, n_samples):
    generated = torch.zeros((imgs.size(0), n_samples, imgs.size(1), imgs.size(2), imgs.size(3)))
    for i in range(n_samples):
        generated[:, i, :, :, :] = vae(imgs, evaluation=True)[0]

    generated = torch.cat(generated.unbind()).unsqueeze(0).squeeze(0)

    return generated"""

# Training
def train(epoch, net, moco_model, optimizer, trainloader, banks):
    loss = 0
    acc = 0
    acc_refined_out = 0
    acc_y_noisy = 0
    perc_f_neg = 0
    perc_f_pos = 0

    net.train()
    moco_model.train()

    num_generated_samples = 1

    for batch_idx, batch in enumerate(trainloader): 
        weak_x = batch[0].cuda()
        strong_x = batch[1].cuda()
        y = batch[2].cuda()
        idxs = batch[3].cuda()
        noisy_y = batch[4].cuda()
        strong_x2 = batch[5].cuda()

        feats_w, logits_w = moco_model(weak_x, cls_only=True)

        if args.label_refinement:
            with torch.no_grad():
                probs_w = F.softmax(logits_w, dim=1)
                pseudo_labels_w, probs_w, pseudo_labels_w_all, hard_pseudo_labels_w = refine_predictions(feats_w, probs_w, banks)
        else:
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w = probs_w.max(1)[1]
        
        _, logits_q, logits_ctr, keys = moco_model(strong_x, strong_x2)

        if args.ctr:
            loss_ctr = contrastive_loss(
                logits_ins=logits_ctr,
                pseudo_labels=moco_model.mem_labels[idxs], #pseudo_labels_w, #,#
                mem_labels=moco_model.mem_labels[moco_model.idxs]
            )
        else:
            loss_ctr = 0
        
        # update key features and corresponding pseudo labels
        moco_model.update_memory(epoch, idxs, keys, pseudo_labels_w, y)

        #FOR ANALYSIS
        false_negative = (moco_model.real_labels[idxs].unsqueeze(1) == moco_model.real_labels[moco_model.idxs].unsqueeze(0)) & (moco_model.mem_labels[idxs,moco_model.mem_ptr].unsqueeze(1) != moco_model.mem_labels[moco_model.idxs,moco_model.mem_ptr].unsqueeze(0))
        false_positive = (moco_model.real_labels[idxs].unsqueeze(1) != moco_model.real_labels[moco_model.idxs].unsqueeze(0)) & (moco_model.mem_labels[idxs,moco_model.mem_ptr].unsqueeze(1) == moco_model.mem_labels[moco_model.idxs,moco_model.mem_ptr].unsqueeze(0))

        wrongly_included_false_negative = false_negative & torch.all(moco_model.mem_labels[idxs].unsqueeze(1) != moco_model.mem_labels[moco_model.idxs].unsqueeze(0), dim=2)
        wrongly_excluded_false_positive = false_positive & (~torch.all(moco_model.mem_labels[idxs].unsqueeze(1) != moco_model.mem_labels[moco_model.idxs].unsqueeze(0), dim=2))

        with torch.no_grad():
            #CE weights
            max_entropy = torch.log2(torch.tensor(args.num_class))
            w = entropy(probs_w)
            #w = (max_entropy - w) / max_entropy
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

        #step = batch_idx + epoch * len(trainloader)
        #adjust_learning_rate(optimizer, step)
        
        accuracy = 100.*accuracy_score(y.to('cpu'), logits_w.to('cpu').max(1)[1])
        accuracy_refined_out = 100.*accuracy_score(y.to('cpu'), pseudo_labels_w.to('cpu'))
        accuracy_y_noisy = 100.*accuracy_score(y.to('cpu'), noisy_y.to('cpu'))

        loss += l.item()
        acc += accuracy
        acc_refined_out += accuracy_refined_out
        acc_y_noisy += accuracy_y_noisy
        perc_f_neg += wrongly_included_false_negative.sum()/false_negative.sum()
        perc_f_pos += wrongly_excluded_false_positive.sum()/false_positive.sum()
        
        if batch_idx % 100 == 0:
            print('Epoch [%3d/%3d] Iter[%3d/%3d]\t ' 
                    %(epoch, args.num_epochs, batch_idx+1, len(trainloader)))

            print("Acc ", acc/(batch_idx+1))
            #print("Acc Y noisy ", acc_y_noisy/(batch_idx+1))
            print("Acc refined Out ", acc_refined_out/(batch_idx+1))
            print(w)

    
    print("Training acc = ", acc/len(trainloader))

    if args.wandb:
        wandb.log({
        'train_loss': loss_cls/len(trainloader), \
        'train_acc': acc/len(trainloader), \
        'constrastive_loss': loss_ctr/len(trainloader), \
        'acc_y_noisy': acc_y_noisy/len(trainloader), \
        'acc_refined_out': acc_refined_out/len(trainloader), \
        'perc_f_neg': perc_f_neg/len(trainloader), \
        'perc_f_neg': perc_f_neg/len(trainloader), \
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
    acc_pre = 100.*accuracy_score(gt_labels.to('cpu'), logits.max(1)[1].to('cpu'))

    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  

    if args.wandb:
        wandb.log({
        'val_accuracy': acc, \
        'val_accuracy_pre': acc_pre, \
        }, step=epoch)
    
    return acc, banks, gt_labels, pred_labels

"""def test(epoch,net):
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
        'val_accuracy': acc, \
        }, step=epoch)
    
    return acc"""

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
    train_dataset = dataset(dataset='mnist', root='./data/', noisy_path=None,
                          mode='all',
                          transform=transforms.Compose([transforms.Resize((36)), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset='mnist', root='./data/', noisy_path=None,
                         mode='all',
                         transform=transforms.Compose([transforms.Resize((36)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset == 'mnistm':
    train_dataset = dataset(dataset='mnistm', root='./data/', noisy_path=None,
                          mode='all',
                          transform=transforms.Compose([transforms.Resize((36)), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset='mnistm', root='./data/', noisy_path=None,
                         mode='all',
                         transform=transforms.Compose([transforms.Resize((36)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset == 'svhn':
    train_dataset = dataset(dataset='svhn', root='./data/', noisy_path=None,
                          mode='all',
                          transform=transforms.Compose([transforms.Resize((36)), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset='svhn', root='./data/', noisy_path=None,
                         mode='all',
                         transform=transforms.Compose([transforms.Resize((36)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset == 'usps':
    train_dataset = dataset(dataset='usps', root='./data/', noisy_path=None,
                          mode='all',
                          transform=transforms.Compose([transforms.Resize((36)), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                          )

    test_dataset = dataset(dataset='usps', root='./data/', noisy_path=None,
                         mode='all',
                         transform=transforms.Compose([transforms.Resize((36)), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                         )

elif args.dataset.split('/')[0] == 'pacs':
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
net = create_model(arch)
momentum_net = create_model(arch)

load_weights(net, 'logs/' + args.source + '/weights_best.tar')
load_weights(momentum_net, 'logs/' + args.source + '/weights_best.tar')

optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4)

#net = nn.DataParallel(net)
#momentum_net = nn.DataParallel(momentum_net)

#moco_model = AdaMoCo(src_model = net, momentum_model = momentum_net, features_length=net.module.bottleneck_dim, num_classes=args.num_class, dataset_length=len(train_dataset))
moco_model = AdaMoCo(src_model = net, momentum_model = momentum_net, features_length=net.bottleneck_dim, num_classes=args.num_class, dataset_length=len(train_dataset), temporal_length=args.temporal_length)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# vae = VAE(128)
#vae = ResNetVAE('resnet18')

#vae.to(device)

#load_weights(vae, args.weights)

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

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
unormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
    )

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
        )
"""
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
#kl_loss = nn.KLDivLoss(reduction="batchmean")

best = 0

acc, banks, _, _ = eval_and_label_dataset(0, moco_model, None)

for epoch in range(args.num_epochs+1):
 
    print('Train Nets')
    train(epoch, net, moco_model, optimizer, train_loader, banks) # train net1  

    #acc = test(epoch,net) 
    acc, banks, gt_labels, pred_labels = eval_and_label_dataset(epoch, moco_model, banks)

    if acc > best:
        save_weights(net, epoch, logdir + '/weights_best.tar')
        np.save(logdir + '/banks_best.npy', banks)
        np.save(logdir + '/gt_labels_best.npy', gt_labels.detach().cpu())
        np.save(logdir + '/pred_best.npy', pred_labels.detach().cpu())
        best = acc
        print("Saving best!")

        if args.wandb:
            wandb.run.summary['best_acc'] = best


