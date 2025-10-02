details......
best_model.h5 – Trained LSTM autoencoder model file for sales anomaly detection.
config.json – Configuration file with model, threshold, and workflow parameters.
features.csv – Weekly economic, weather, markdown, and holiday features.
main.py – Main script to merge data, train model, detect anomalies, and output results.
run_stats.json – Run metrics and the anomaly threshold used for detection.
store_20_anomaly.png – Anomaly error plot for Store 20 (visual output).
store_24_anomaly.png – Anomaly error plot for Store 24 (visual output).
store_39_anomaly.png – Anomaly error plot for Store 39 (visual output).
stores.csv – Information about each store (type, size, and number).
test.csv – Kaggle test file (for out-of-sample forecasts, not used in anomaly pipeline).
train.csv – Kaggle weekly sales history per store/department (core input for model).
results.csv - result
