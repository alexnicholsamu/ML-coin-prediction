import train_model
import visualize

# Visualize the results
if __name__ == "__main__":
    data_pack = {"coin": "Bitcoin",
                 "learning rate": 3e-2,
                 "number_epochs": 150,
                 "patience": 10}
    actual, predicted, predicted_std, tomorrow_price, mse = train_model.getPredictions(data_pack)  # Change for desired coin
    print(mse)
    print("Tomorrow's predicted price: $" + str(tomorrow_price))
    visualize.getPlot(actual, predicted, predicted_std)