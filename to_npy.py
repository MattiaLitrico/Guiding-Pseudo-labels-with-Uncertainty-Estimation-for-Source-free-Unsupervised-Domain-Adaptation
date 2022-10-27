import numpy as np
import argparse
import glob
import os
import warnings
import sys
from PIL import Image
from datasets import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--dataset', type=str)
parser.add_argument('--mode', type=str)

args = parser.parse_args()
warnings.filterwarnings("ignore")

if args.dataset.split('/')[0] == 'pacs':
    dataset = dataset(dataset=args.dataset, root='./data/PACS',
                          mode=args.mode,
                          transform=None)
elif args.dataset.split('/')[0] == 'visdac':
    dataset = dataset(dataset=args.dataset, root='./data/VisdaC',
                          mode=args.mode,
                          transform=None)
elif args.dataset.split('/')[0] == 'domainnet':
    dataset = dataset(dataset=args.dataset, root='./data/domainnet',
                          mode=args.mode,
                          transform=None)

imgs = []
labels = []

for i, data in enumerate(dataset):
    print("Processing img {}/{}".format(i, len(dataset)))
    imgs.append(data[0])
    labels.append(data[2])

imgs = np.stack(imgs, axis=0)
labels = np.array(labels)

if args.dataset.split('/')[0] == 'pacs':
    np.save(os.path.join("data","PACS",args.dataset.split('/')[1]+"_"+args.mode+"_imgs.npy"), imgs)
    np.save(os.path.join("data","PACS",args.dataset.split('/')[1]+"_"+args.mode+"_labels.npy"), labels)
elif args.dataset.split('/')[0] == 'visdac':
    np.save(os.path.join("data","VisdaC",args.dataset.split('/')[1]+"_"+args.mode+"_imgs.npy"), imgs)
    np.save(os.path.join("data","VisdaC",args.dataset.split('/')[1]+"_"+args.mode+"_labels.npy"), labels)
elif args.dataset.split('/')[0] == 'domainnet':
    np.save(os.path.join("data","domainnet",args.dataset.split('/')[1]+"_imgs.npy"), imgs)
    np.save(os.path.join("data","domainnet",args.dataset.split('/')[1]+"_labels.npy"), labels)