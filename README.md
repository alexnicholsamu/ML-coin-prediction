# ML-coin-prediction, a cryptocurrency prediction neural network by Alexander Nichols

## Tools used in the creation of this project:

> [CoinGeckoAPI](https://www.coingecko.com/en/api)

> PyTorch, sklearn, matplotlib, Pandas, NumPy

## Summary:

This is my coin prediction algorithm. This is the alpha stage, so there will be improvements to come. Soon to be implemented: accuracy improvements and better visuals

I use a Deep Feed-Forward Neural Network model (with a recurrent layer) with a resilient backpropagation optimizer that I train using cryptocurrency data from a certain (editable in [data_prep.py](data_prep.py)) start date; it trains over the trends of the price, volatility, and RSI to generate a prediction for tomorrow's price (along with a visual demonstrating it's 'thought process' overtime). 

The model trains itself over a fake, generated cryptocurrency that it pits against the real price history, as the algorithm trains itself based on the correctness of the fake coin relative to the real one. As previously stated, the model takes into account the price history, volatility, and RSI to understand the movements of the coin and generate its tomorrow prediction.

Coins available: All coins available in the CoinGeckoAPI seen [here](https://www.coingecko.com/en/all-cryptocurrencies)

All files are necessary to run this and they should be run through [model_run.py](model_run.py). The license can be found [here](LICENSE), and any and all suggestions should be emailed to _alexander.k.nichols@gmail.com_

Sample Image given example data batch (_29/04/23_):

![Sample Image](./images/sampleimage.png)
