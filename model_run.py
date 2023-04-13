import numpy as np
import matplotlib.pyplot as plt
import train_model

# Visualize the results
if __name__ == "__main__":
    actual, predicted, predicted_std = train_model.getPredictions("Bitcoin")  # Change for desired coin
    plt.figure(figsize=(16, 6))
    plt.plot(actual, label='Actual Prices')
    plt.plot(predicted, label='Predicted Prices')
    plt.fill_between(np.arange(len(predicted)),
                     predicted.squeeze() - predicted_std.squeeze(),
                     predicted.squeeze() + predicted_std.squeeze(),
                     alpha=0.5,
                     label='Uncertainty')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
