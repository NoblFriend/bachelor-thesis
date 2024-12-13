import torch
from torch.func import functional_call

def mirror_descent(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    data_loader: torch.utils.data.DataLoader,
    param_name: str,
    impact: torch.Tensor,
    model_lr: float,
    md_lr: float,
    md_lambda: float,
    md_num_steps: int,
    criterion: torch.nn.Module,
) -> torch.Tensor:

    impact = impact.clone().detach().requires_grad_(True)
    original_param = dict(model.named_parameters())[param_name]

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    param_grad = torch.autograd.grad(loss, original_param, create_graph=True)[0]
    new_params = {name: param.clone() for name, param in model.named_parameters()}

    for _ in range(md_num_steps):
        param_new = original_param - md_lr * impact * param_grad
        new_params[param_name] = param_new
        outputs_new = functional_call(model, new_params, (X_train,))
        loss_new = criterion(outputs_new, y_train)

        grad_impact = torch.autograd.grad(loss_new, impact)[0]

        with torch.no_grad():
            impact_update = torch.pow(impact * torch.exp(-md_lr * grad_impact), 1/(1+md_lr*md_lambda))
            impact = impact_update / impact_update.sum()

        impact.requires_grad_(True)

    return impact.detach()
