from utils.zero_grad import zero_out_gradients

def step(X_batch, y_batch, model, criterion, optimizer, impacts, drop_ratio, random_mode, drop_connections, device):
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)

    optimizer.zero_grad()
    loss.backward()
    if drop_connections:
        zero_out_gradients(model, impacts, share=drop_ratio, random=random_mode)
    return loss.item()