import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os, json

# --- Load config ---
with open("config.json") as f:
    config = json.load(f)

np.random.seed(config["RANDOM_SEED"])
os.makedirs(config["RESULTS_DIR"], exist_ok=True)

train = pd.read_csv("train.csv")
features = pd.read_csv("features.csv")
stores = pd.read_csv("stores.csv")

data = train.merge(features, on=["Store", "Date"], how="left").merge(stores, on="Store", how="left")
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
data = data.sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)

# --- Ensure IsHoliday column is present as integer 0/1
iso_cols = [c for c in data.columns if "IsHoliday" in c]
if len(iso_cols) > 1:
    # Merge any IsHoliday columns using "OR" logic for robustness
    data["IsHoliday"] = data[iso_cols].apply(lambda x: int(any(
        v in [True, 1, "TRUE", "True"] for v in x)), axis=1)
elif "IsHoliday" in data.columns:
    data["IsHoliday"] = data["IsHoliday"].apply(lambda x: 1 if x in [True, 1, "TRUE", "True"] else 0)
else:
    raise ValueError("IsHoliday not found in the merged data.")

# --- Features setup ---
markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
for col in markdown_cols:
    if col in data.columns:
        data[col] = data[col].fillna(0)

base_features = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday"]
feature_cols = [c for c in base_features + markdown_cols if c in data.columns]
print("Using feature columns:", feature_cols)
scaler = MinMaxScaler()
data[feature_cols] = scaler.fit_transform(data[feature_cols])

def create_sequences(data, seq_length, feature_cols):
    sequences, meta = [], []
    for store in data.Store.unique():
        store_data = data[data.Store == store]
        for dept in store_data.Dept.unique():
            dept_data = store_data[store_data.Dept == dept].reset_index(drop=True)
            if len(dept_data) < seq_length:
                continue
            for i in range(len(dept_data) - seq_length + 1):
                seq = dept_data[feature_cols].iloc[i:i+seq_length].values
                sequences.append(seq)
                meta.append({
                    "Store": store,
                    "Dept": dept,
                    "Start_Date": str(dept_data.Date.iloc[i].date()),
                    "End_Date": str(dept_data.Date.iloc[i+seq_length-1].date())
                })
    return np.array(sequences), meta

X, meta = create_sequences(data, config["SEQUENCE_LENGTH"], feature_cols)
print(f"Global sequences shape: {X.shape}")

def build_model(seq_length, n_features, config):
    inp = Input(shape=(seq_length, n_features))
    x = LSTM(config["LSTM_UNITS"], activation="relu", return_sequences=False)(inp)
    x = Dropout(config["DROPOUT_RATE"])(x)
    x = Dense(config["DENSE_UNITS"], activation="relu")(x)
    x = RepeatVector(seq_length)(x)
    x = LSTM(config["LSTM_UNITS"], activation="relu", return_sequences=True)(x)
    x = Dropout(config["DROPOUT_RATE"])(x)
    out = TimeDistributed(Dense(n_features))(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

n_features = X.shape[2]
autoencoder = build_model(config["SEQUENCE_LENGTH"], n_features, config)
autoencoder.summary()

earlystop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    os.path.join(config["RESULTS_DIR"], "best_model.h5"),
    monitor="loss", save_best_only=True, verbose=1
)
history = autoencoder.fit(
    X, X,
    epochs=config["EPOCHS"],
    batch_size=config["BATCH_SIZE"],
    shuffle=True,
    callbacks=[earlystop, checkpoint]
)

X_pred = autoencoder.predict(X, verbose=0)
mse = np.mean(np.power(X - X_pred, 2), axis=(1,2))

# --- Stats and Threshold ---
stats = {
    "mean_reconstruction_error": float(np.mean(mse)),
    "std_reconstruction_error": float(np.std(mse)),
    "min_reconstruction_error": float(np.min(mse)),
    "max_reconstruction_error": float(np.max(mse)),
    "final_loss": float(history.history["loss"][-1]),
}
if config["THRESHOLD_TYPE"] == "global":
    threshold = stats["mean_reconstruction_error"] + config["THRESHOLD_KSTD"] * stats["std_reconstruction_error"]
elif config["THRESHOLD_TYPE"] == "percentile":
    threshold = np.percentile(mse, config["THRESHOLD_PERCENTILE"])
else:
    raise ValueError("Invalid threshold type!")

stats["anomaly_threshold"] = float(threshold)
with open(os.path.join(config["RESULTS_DIR"], "run_stats.json"), "w") as outf:
    json.dump(stats, outf, indent=2)
print(f"Run stats written to {os.path.join(config['RESULTS_DIR'], 'run_stats.json')}")

# --- Anomaly Report ---
anomaly_report = []
for idx in range(len(meta)):
    entry = meta[idx].copy()
    entry["Reconstruction_Error"] = mse[idx]
    entry["Threshold"] = threshold
    entry["Is_Anomaly"] = "Yes" if mse[idx] > threshold else "No"

    seq_true = X[idx]
    seq_pred = X_pred[idx]
    feature_errors = np.mean(np.abs(seq_true - seq_pred), axis=0)
    feature_deviation = dict(zip(feature_cols, feature_errors))
    max_feature = max(feature_deviation, key=feature_deviation.get)
    is_holiday = False
    if "IsHoliday" in feature_cols:
        col_idx = feature_cols.index("IsHoliday")
        is_holiday = any(seq_true[:, col_idx] == 1)
    reason = f"Largest error: {max_feature}"
    if is_holiday:
        reason += "; IsHoliday: True"

    if "Weekly_Sales" in feature_cols:
        sales_idx = feature_cols.index("Weekly_Sales")
        actual_sales = np.mean(seq_true[:, sales_idx])
        pred_sales = np.mean(seq_pred[:, sales_idx])
        if actual_sales > pred_sales:
            highlow = "High Sale"
        elif actual_sales < pred_sales:
            highlow = "Low Sale"
        else:
            highlow = "Neutral"
    else:
        highlow = ""
    entry["Sale_Anomaly_Type"] = highlow

    entry["Reason"] = reason
    anomaly_report.append(entry)

anomaly_df = pd.DataFrame(anomaly_report)
anomaly_df = anomaly_df.sort_values(by=["Store", "Dept", "Start_Date"]).reset_index(drop=True)
anomaly_report_path = os.path.join(config["RESULTS_DIR"], "results.csv")
anomaly_df.to_csv(anomaly_report_path, index=False)
print(f"Sorted, annotated anomaly report saved as {anomaly_report_path}")

# --- PLOTS FOR TOP 3 ANOMALY-RICH STORES ---
top_stores = anomaly_df[anomaly_df["Is_Anomaly"]=="Yes"]["Store"].value_counts().head(3).index.tolist()
for store in top_stores:
    store_df = anomaly_df[(anomaly_df["Store"]==store) & (anomaly_df["Is_Anomaly"]=="Yes")]
    plt.figure(figsize=(12,6))
    plt.plot(store_df["Start_Date"], store_df["Reconstruction_Error"], 'r*-')
    plt.title(f'Anomaly Reconstruction Error for Store {store}')
    plt.xlabel('Start Date')
    plt.ylabel('Reconstruction Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_path = os.path.join(config["RESULTS_DIR"], f"store_{store}_anomaly.png")
    plt.savefig(img_path)
    plt.close()

print(f"Anomaly plots saved in {config['RESULTS_DIR']}")
