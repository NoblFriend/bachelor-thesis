import torch
from tqdm import tqdm
def update_accuracy(model, trainloader, testloader, device):
    accuracy = {"train": 0.0, "test": 0.0}
    loaders = {"train": trainloader, "test": testloader}
    total = {"train": 0, "test": 0}

    model.eval()
    with torch.no_grad():
        for loader_type, loader in loaders.items():
            for data in tqdm(loader, desc=f"Updating {loader_type} accuracy", leave=False, ncols=100):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted_train = torch.max(outputs, 1)
                total[loader_type] += labels.size(0)
                accuracy[loader_type] += (predicted_train == labels).sum().item()

    train_accuracy = accuracy["train"] / total["train"]
    test_accuracy = accuracy["test"] / total["test"]
    return train_accuracy, test_accuracy