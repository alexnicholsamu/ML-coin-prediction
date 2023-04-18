import train_model
import visualize

# Visualize the results
if __name__ == "__main__":
    actual, predicted, predicted_std = train_model.getPredictions("Monero")  # Change for desired coin
    visualize.getPlot(actual, predicted, predicted_std)
