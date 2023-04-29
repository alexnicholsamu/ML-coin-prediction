# ML-coin-prediction, a cryptocurrency prediction neural network by Alexander Nichols

## Tools used in the creation of this project:

> [Dataset](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory)
 
> PyTorch, sklearn, pycoingecko

## Summary:

This is my coin prediction algorithm. This is the alpha stage, so there will be improvements to come. Soon to be implemented: accuracy improvements and better visuals

I use a Deep Feed-Forward Neural Network (DFFNN) model with a resilient backpropagation optimizer that I train using the dataset above and pipeline-ing in the most current data; it trains over the trends of the price following it's own conceptual "fake" cryptocurrency running cocurrently. Then, the coin predicted history vs actual history and a prediction for tomorrows price is given.

Coins available: Aave, BinanceCoin, Bitcoin, Cardano, ChainLink, Cosmos, CryptocomCoin, Dogecoin, EOS, Ethereum, Iota, Litecoin, Monero, NEM, Polkadot, Solana, Stellar, Tether, Tron, Uniswap, USDCoin, WrappedBitcoin, XRP

All files are necessary to run this

License can be found in the [License](LICENSE), and any and all suggestions should be emailed to _alexander.k.nichols@gmail.com_

Sample Image given example data batch:

![Sample Image](./images/sampleimage.png)