import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pyro
from pyro.infer.autoguide import AutoDiagonalNormal
import model_architecture


def chooseData(coin):
    data = pd.read_csv(f'coin_{coin}.csv')
    # Keep only the 'Close' price column
    price_data = data['Close']

    # Normalize the data (useful for LSTM)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price_data_normalized = scaler.fit_transform(price_data.values.reshape(-1, 1))
    return scaler, price_data_normalized


model = model_architecture.getModel()
guide = AutoDiagonalNormal(model)

svi = pyro.infer.SVI(model=model,
                     guide=guide,
                     optim=pyro.optim.Adam({"lr": 5e-5}),
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


def sortData(normalized_data):
    seq_length = 30
    inputs, labels = create_sequences(normalized_data, seq_length)

    # Split the data into train and test sets
    train_size = int(len(inputs) * 0.8)
    train_inputs, train_labels = inputs[:train_size], labels[:train_size]
    test_inputs, test_labels = inputs[train_size:], labels[train_size:]

    # Convert data to PyTorch tensors
    train_inputs = torch.tensor(train_inputs).float()
    train_labels = torch.tensor(train_labels).float()
    test_inputs = torch.tensor(test_inputs).float()
    test_labels = torch.tensor(test_labels).float()

    return train_inputs, train_labels, test_inputs, test_labels


def training(num_epochs, train_inputs, train_labels):
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(len(train_inputs)):
            inputs = train_inputs[i].unsqueeze(0)  # Add a new dimension for the sequence
            labels = train_labels[i].unsqueeze(0)  # Add a new dimension for the sequence
            loss = svi.step(inputs, labels)
            total_loss += loss
        avg_loss = total_loss / len(train_inputs)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.2f}')


def getMeanSquaredError(predicted, actual):
    mse = np.mean((predicted - actual) ** 2)
    return f"Mean Squared Error: {mse:.2f}"


def getPredictions(coin):
    scaler, normalized_data = chooseData(coin)
    train_inputs, train_labels, test_inputs, test_labels = sortData(normalized_data)
    num_epochs = 50
    training(num_epochs, train_inputs, train_labels)

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

# Calculate the mean squared error

