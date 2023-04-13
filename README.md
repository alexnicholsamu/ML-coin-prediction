# monero-coin-prediction

[Dataset Used](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory?select=coin_Monero.csv)

This is my Monero coin prediction algorithm. This is the alpha stage, so there will be improvements to come. Soon to be implemented: data for coins beyond Monero, pipeline to get more data, and more! (Ideally, a better architecture as well)

I use a BayesianLSTM model that I train using the dataset above, that predicts future prices of Monero using market patterns from coin history. Then, the training history is plotted. 

Code can be found in the [Lisense](LICENSE), and any and all suggestions should be emailed to _alexander.k.nichols@gmail.com_