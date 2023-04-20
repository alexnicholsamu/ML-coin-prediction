import train_model
import visualize

# Visualize the results
if __name__ == "__main__":
    learning_rate = 5e-5
    num_epochs = 150
    actual, predicted, predicted_std, tomorrow_price = train_model.getPredictions("Dogecoin", learning_rate, num_epochs)  # Change for desired coin
    print("Tomorrow's predicted price: $" + str(tomorrow_price))
    visualize.getPlot(actual, predicted, predicted_std)