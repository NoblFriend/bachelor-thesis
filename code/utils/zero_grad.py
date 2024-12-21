import torch

def zero_out_gradients(model, impacts, share=0.25, random=False):
    if share == 1:
        return
    for name, param in model.named_parameters():
        if name in impacts:
            importance = impacts[name].view(-1)
            num_elements = importance.numel()
            num_to_keep = max(int(num_elements * share), 1)
            
            if random:
                probabilities = (p:= torch.ones_like(importance)) / p.sum()
            else:
                probabilities = importance / importance.sum() 
    
            keep_indices = torch.multinomial(probabilities, num_to_keep, replacement=False)

            mask = torch.zeros_like(param.grad.view(-1), dtype=torch.bool)
            mask[keep_indices] = True
            
            param.grad *= mask.view(param.shape)
