import os
import os.path
import torch

def save_weights(model, e, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    
    torch.save({
            'epochs': e,
            'weights': model.state_dict()}, filename)

def load_checkpoint(weights, cpu=False):
    if not cpu:
        checkpoint = torch.load(weights)
    else:
        checkpoint = torch.load(weights, map_location=torch.device('cpu'))
    
    return checkpoint

def load_weights(model, weights):
    checkpoint = load_checkpoint(weights)
    model.load_state_dict(checkpoint['weights'])

    return model
