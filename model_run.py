import train_model
import visualize

# Visualize the results
if __name__ == "__main__":
    data_pack = {"learning rate": 5e-5,
                 "number_epochs": 150,
                 "patience": 10}
    actual, predicted, predicted_std, tomorrow_price = train_model.getPredictions("Monero", data_pack)  # Change for desired coin
    print("Tomorrow's predicted price: $" + str(tomorrow_price))
    visualize.getPlot(actual, predicted, predicted_std)