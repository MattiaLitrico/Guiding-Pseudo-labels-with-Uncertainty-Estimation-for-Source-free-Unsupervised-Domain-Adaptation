
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet101

class Resnet(nn.Module):
    def __init__(self, arch, args):
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