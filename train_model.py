import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import model_architecture
import data_prep

model = model_architecture.getModel()


def training(num_epochs, train_inputs, train_labels, patience, learn_rate, batch_size=128):
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Rprop(model.parameters(), lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.1)
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.mean((outputs.squeeze(-1) - labels) ** 2)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_inputs)
        scheduler.step(avg_loss)
        print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss}')


def getMeanSquaredError(predicted, actual):
    mse = np.mean((predicted - actual) ** 2)
    return f"Mean Squared Error: {mse:.2f}"


def getPredictions(data_pack):
    scaler, normalized_data = data_prep.chooseData(data_pack["coin"])
    train_inputs, train_labels,  test_inputs, test_labels = data_prep.sortData(normalized_data)
    training(data_pack["number_epochs"], train_inputs, train_labels, data_pack["patience"], data_pack["learning rate"])

    with torch.no_grad():
        predicted = model(test_inputs).numpy()

    actual = scaler.inverse_transform(test_labels.numpy().reshape(-1, 1))
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1))

    # Calculate the standard deviation of the predicted values
    predicted_std = np.std(predicted)
    print(f"Mean Squared Error: {getMeanSquaredError(predicted, actual)}")

    tomorrow_price = predicted[len(actual)-1][0]

    return actual, predicted, predicted_std, data_prep.prep_tomorrow_price(tomorrow_price)
