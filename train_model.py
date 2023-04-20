import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pyro
from pyro.infer.autoguide import AutoDiagonalNormal
import model_architecture
import data_prep


model = model_architecture.getModel()
guide = AutoDiagonalNormal(model)

svi = pyro.infer.SVI(model=model,
                     guide=guide,
                     optim=pyro.optim.Adam({"lr": 4e-5}),
                     loss=pyro.infer.Trace_ELBO())



def training(num_epochs, train_inputs, train_labels, val_inputs, val_labels, batch_size=32):
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    patience = 10
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_dataloader:
            loss = svi.step(inputs, labels)
            total_loss += loss

        avg_loss = total_loss / len(train_inputs)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.2f}')

        # Evaluate on validation set
        val_loss = evaluate(val_inputs, val_labels)
        print(f'Validation Loss: {val_loss:.2f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

def evaluate(inputs, labels):
    with torch.no_grad():
        loss = svi.evaluate_loss(inputs, labels)
    return loss


def getMeanSquaredError(predicted, actual):
    mse = np.mean((predicted - actual) ** 2)
    return f"Mean Squared Error: {mse:.2f}"


def getPredictions(coin):
    scaler, normalized_data = data_prep.chooseData(coin)
    train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, last_sequence = data_prep.sortData(normalized_data)
    num_epochs = 150
    training(num_epochs, train_inputs, train_labels, val_inputs, val_labels)

    with torch.no_grad():
        predictive = pyro.infer.Predictive(model_architecture.getModel(), guide=guide, num_samples=1000)
        samples = predictive(test_inputs)
        predicted = samples['obs'].mean(0).detach().numpy()
        predicted_std = samples['obs'].std(0).detach().numpy()

    predicted = scaler.inverse_transform(predicted)
    predicted_std = scaler.inverse_transform(predicted_std)
    actual = scaler.inverse_transform(test_labels.numpy().reshape(-1, 1))

    print(getMeanSquaredError(predicted, actual))

    # Make a prediction for tomorrow
    with torch.no_grad():
        tomorrow_normalized_samples = predictive(last_sequence)['obs']
        tomorrow_normalized_mean = tomorrow_normalized_samples.mean(0).item()
        tomorrow_price = scaler.inverse_transform(np.array(tomorrow_normalized_mean).reshape(1, -1))

    tomorrow_price = tomorrow_price[0, 0]

    return actual, predicted, predicted_std, tomorrow_price