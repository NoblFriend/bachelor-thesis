import torch

def is_layer_with_impact(name):
    return "bn" not in name and "shortcut" not in name and "linear" not in name

def generate_impacts(model, func, device):
    return {
        name: (imp := func(param)).to(device) / imp.sum()
        for name, param in model.named_parameters()
        if is_layer_with_impact(name)
    }

def initialize_impacts(model, device, ones):
    if ones:
        return generate_impacts(model, torch.ones_like, device)
    else:
        return generate_impacts(model, lambda param: param.grad.detach()**2, device)

