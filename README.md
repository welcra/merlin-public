# merlin-public
**This is the public version of a private repository, several files were removed**

Files removed:
- crypto\crypto_bot.py (real time crypto trading bot)
- crypto\crypto_test.py (used for testing parts of the bot)
- forex\forex_data (forex data)
- forex\forex_data_1m (forex data with a 1m interval)
- Markowitz Optimization\stock_data (ohlcv data)
- Markowitz Optimization\data.py (get ohlcv data)
- Neural Network\altman_only (altman z-scores at the time of stocks rising 20%)
- Neural Network\stock_data (ohlcv data with a 1d interval for most stocks listed on nasdaq with the max period)
- Neural Network\v1 (version 1 of company information at 20% jumps)
- Neural Network\v2 (version 2 of company information at 20% jumps)
- Neural Network\v3 (version 3 of company information at 20% jumps)
- Neural Network\historical_buys_1mo.csv (list of all times stocks jumped 20% in a month)
- Neural Network\historical_buys_1wk.csv (list of all times stocks jumped 20% in a week)
- Neural Network\historical_sells_1mo.csv (list of all times stocks went down 20% in a month)
- Neural Network\historical_sells_1wk.csv (list of all times stocks went down 20% in a week)
- Neural Network\historical_data.py (gets stock ohlcv data for stock_data)
- Neural Network\metrics_at_buys.py (gets company information at each historical buy)
- Neural Network\metrics_at_sells.py (gets company information at each historical sell)
- Neural Network\model_xgb_1mo.json (xgboost model for predicting 1mo jumps)
- Neural Network\model_xgb_1wk.json (xgboost model for predicting 1wk jumps)
- Neural Network\model_xgb.json (older xgboost model, not used)
- Neural Network\model.keras (feedforward neural network model for predicting 1wk jumps, not used because of an error in creation)
- Neural Network\test_feedforward.py (testing the feedforward neural network)
- Neural Network\test_model.py (testing the xgboost models (get the scores in scores_1mo.csv and scores_1wk.csv))

Projects related to quantitative trading and finance

Types of models & systems:

- Identify stocks with highest dividend yields
- Portfolio optimizer (total stock market) using the markowitz model in modern portfolio theory
- Creating portfolios using sharpe ratios
- Valuation model using metrics like P/E ratio, P/B ratio, altman z-score, etc
- Algorithmic trader using vectorbt (unfinished and will probably stay that way, might create another one custom made without vectorbt)
- Neural Network for identifying events of weekly gain (unfinished, current project)

Notes for neural network [HERE](https://github.com/Arnav-MaIhotra/merlin/blob/main/Neural%20Network/neural_network_notes.md)

Notes for forex [HERE](https://github.com/Arnav-MaIhotra/merlin/blob/main/forex/forex_notes.md)

Notes for crypto [HERE](https://github.com/Arnav-MaIhotra/merlin/blob/main/crypto/crypto_notes.md)