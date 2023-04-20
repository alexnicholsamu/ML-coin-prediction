import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pyro
from pyro.infer.autoguide import AutoDiagonalNormal
import model_architecture


def chooseData(coin):
    data = pd.read_csv(f'data_csv/coin_{coin}.csv')
    # Keep only the 'Close' price column
    price_data = data['Close'].to_numpy()

    # Normalize the data (useful for LSTM)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price_data_normalized = scaler.fit_transform(price_data.reshape(-1, 1))
    return scaler, price_data_normalized



model = model_architecture.getModel()
guide = AutoDiagonalNormal(model)

svi = pyro.infer.SVI(model=model,
                     guide=guide,
                     optim=pyro.optim.Adam({"lr": 4e-5}),
                     loss=pyro.infer.Trace_ELBO())


def create_sequences(data, seq_length):
    inputs = []
    labels = []

    for i in range(len(data) - seq_length):
        inputs.append(data[i:i + seq_length, 0])
        labels.append(data[i + seq_length])

    inputs = np.array(inputs).reshape(-1, seq_length, 1)
    labels = np.array(labels)

    return inputs, labels


def sortData(data, train_ratio=0.7, val_ratio=0.2):
    seq_length = 30
    inputs, labels = create_sequences(data, seq_length)
    train_size = int(len(inputs) * train_ratio)
    val_size = int(len(inputs) * val_ratio)

    train_inputs = torch.tensor(inputs[:train_size], dtype=torch.float32)
    train_labels = torch.tensor(labels[:train_size], dtype=torch.float32)

    val_inputs = torch.tensor(inputs[train_size:train_size + val_size], dtype=torch.float32)
    val_labels = torch.tensor(labels[train_size:train_size + val_size], dtype=torch.float32)

    test_inputs = torch.tensor(inputs[train_size + val_size:], dtype=torch.float32)
    test_labels = torch.tensor(labels[train_size + val_size:], dtype=torch.float32)

    return train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels



def training(num_epochs, train_inputs, train_labels, val_inputs, val_labels, batch_size=32):
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    patience = 5
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
    scaler, normalized_data = chooseData(coin)
    train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = sortData(normalized_data)
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

    return actual, predicted, predicted_std