import numpy as np
import matplotlib.pyplot as plt


def getPlot(actual, predicted, predicted_std):
    plt.figure(figsize=(16, 6))
    plt.plot(actual, label='Actual Prices')
    plt.plot(predicted, label='Predicted Prices')
    plt.fill_between(np.arange(len(predicted)),
                     predicted.squeeze() - predicted_std.squeeze(),
                     predicted.squeeze() + predicted_std.squeeze(),
                     alpha=0.25,
                     label='Uncertainty')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
