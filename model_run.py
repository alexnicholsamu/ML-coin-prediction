import train_model
import visualize

# Visualize the results
if __name__ == "__main__":
    actual, predicted, predicted_std, tomorrow_price = train_model.getPredictions("Monero")  # Change for desired coin
    print("Tomorrow's predicted price: $" + str(round(tomorrow_price, 2)))
    visualize.getPlot(actual, predicted, predicted_std)
