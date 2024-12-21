def training_cycle(trainloader, testloader, num_epochs, update_epochs, share, learning_rate=0.01, lambda_value=1, plot_histograms=False, random_mode=False, use_impacts=True):
    set_seed()
    model = ResNet18().to(DEVICE)
    model.train()

    impacts = initialize_impacts(model, ones=True)

    log_dict = {
        "impacts": [],
        "train_accuracies": [],
        "test_accuracies": [],
        "losses": []
    }

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}...")
        with tqdm(total=len(trainloader), desc='Training Progress') as pbar:
            for X_batch, y_batch in trainloader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                if use_impacts:
                    if random_mode:
                        random_zero_out_gradients(model, impacts, share=share)
                    else:
                        zero_out_gradients(model, impacts, share=share)

                optimizer.step()
                # TODO: running average loss
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}'
                })
                pbar.update(1)

        if epoch % update_epochs == 0:
            if use_impacts:
                print("Updating impacts...")

                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()

                impacts = initialize_impacts(model, ones=False)

                for name in impacts.keys():
                    impacts[name] = mirror_descent(
                        model=model,
                        param_name=name,
                        impact=impacts[name],
                        lr=learning_rate,
                        eta=2,
                        lambda_value=lambda_value,
                        num_steps=50,
                        X_train=X_batch,
                        y_train=y_batch,
                        criterion=criterion
                    )
                log_dict["impacts"].append(impacts)

                if plot_histograms:
                    plot_impacts_histograms(impacts, share)
            
            print("Updating accuracy...")
            train_accuracy, test_accuracy = update_accuracy(model, trainloader, testloader, device=DEVICE)
            log_dict["train_accuracies"].append(train_accuracy)
            log_dict["test_accuracies"].append(test_accuracy)
            log_dict["losses"].append(loss.item())
            print("Loss: {:.4f} \t Train accuracy: {:.4f} \t Test accuracy: {:.4f}".format(loss.item(), train_accuracy, test_accuracy))


    return log_dict