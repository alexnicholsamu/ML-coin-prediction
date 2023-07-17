import train_model
import visualize

# Visualize the results
if __name__ == "__main__":
    data_pack = {"coin": "Bitcoin",  # Change for desired coin
                 "learning rate": 0.01,
                 "number_epochs": 2500,
                 "patience": 60}
    actual, predicted, predicted_std, tomorrow_price = train_model.getPredictions(data_pack)  
    print("Tomorrow's predicted price: $" + str(tomorrow_price))
    visualize.getPlot(actual, predicted, predicted_std)
