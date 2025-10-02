## 📂 Repository Structure

- **best_model.h5** – Trained LSTM autoencoder model used for anomaly detection  
- **config.json** – Configuration file containing model parameters, thresholds, and workflow settings  
- **features.csv** – Weekly external features (economic, weather, markdowns, holidays)  
- **main.py** – Main pipeline script: merges data, trains model, detects anomalies, and outputs results  
- **run_stats.json** – Run metrics and final anomaly threshold applied  
- **results.csv** – Final anomaly detection results across stores  
- **store_20_anomaly.png** – Anomaly error plot for Store 20  
- **store_24_anomaly.png** – Anomaly error plot for Store 24  
- **store_39_anomaly.png** – Anomaly error plot for Store 39  
- **stores.csv** – Store metadata (type, size, number)  
- **train.csv** – Kaggle weekly sales history per store/department (core dataset for training)  
- **test.csv** – Kaggle test file for out-of-sample forecasts (**not used** in anomaly pipeline)  

---

## 🚀 Workflow

1. **Data Preparation**  
   - Merge `train.csv`, `stores.csv`, and `features.csv`  
   - Normalize features for training  

2. **Model Training**  
   - LSTM Autoencoder is trained to reconstruct normal sales patterns  
   - Reconstruction error is calculated  

3. **Anomaly Detection**  
   - A threshold (from `config.json`) is applied on reconstruction error  
   - Points above threshold are flagged as anomalies  

4. **Outputs**  
   - `results.csv` → anomaly flags per store-week  
   - `run_stats.json` → run metadata and chosen threshold  
   - Plots (`store_20_anomaly.png`, `store_24_anomaly.png`, `store_39_anomaly.png`)  

