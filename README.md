# Time-Series Anomaly Detection (Pump Sensor Data)

Unsupervised anomaly detection on industrial pump sensor data, using the Kaggle
[Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) dataset
(53 real sensors, ~220,000 readings, with 7 recorded real failure events).

The goal: detect abnormal sensor behavior that precedes known pump failures, without
being told in advance where the failures are. Two approaches are built and compared:

1. **Isolation Forest** (scikit-learn) on rolling mean/std features — the baseline.
2. **LSTM Autoencoder** (PyTorch) trained on sequences of normal operation, flagging
   high reconstruction error as anomalous — the stretch goal.

## Approach

### 1. Data loading & cleaning

Sensor columns with >30% missing values are dropped; remaining gaps are forward/backward
filled. `machine_status` (`NORMAL` / `BROKEN` / `RECOVERING`) is kept only as ground truth
for evaluation — it is never used as a training signal, since this is unsupervised.

```python
import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("nphantawee/pump-sensor-data")
df = pd.read_csv(os.path.join(path, "sensor.csv"), parse_dates=["timestamp"])
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

null_frac = df.isnull().mean()
df = df.drop(columns=null_frac[null_frac > 0.3].index.tolist())

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
df[sensor_cols] = df[sensor_cols].fillna(method="ffill").fillna(method="bfill")

failure_times = df.loc[df["machine_status"] == "BROKEN", "timestamp"]
```

### 2. Isolation Forest baseline

Each row is expanded with a rolling mean and rolling standard deviation (60-step window)
per sensor, so the model sees short-term trend, not just the instantaneous reading.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

window = 60
feat_df = df[sensor_cols].copy()
for col in sensor_cols:
    feat_df[f"{col}_rollmean"] = df[col].rolling(window).mean()
    feat_df[f"{col}_rollstd"] = df[col].rolling(window).std()
feat_df = feat_df.dropna().reset_index(drop=True)
meta = df.iloc[window-1:].reset_index(drop=True)

X = StandardScaler().fit_transform(feat_df)
iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
iso.fit(X)

results = meta[["timestamp", "machine_status"]].copy()
results["iso_score"] = -iso.decision_function(X)
results["iso_anomaly"] = iso.predict(X) == -1
```

### 3. LSTM Autoencoder (stretch goal)

Trained only on `NORMAL`-labeled stretches, on sequences of 30 consecutive timesteps.
Anomaly score = reconstruction error against the full dataset (batched to avoid GPU OOM
on ~220k sequences at once).

```python
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

seq_len = 30
normal_scaled = StandardScaler().fit_transform(df[df["machine_status"] == "NORMAL"][sensor_cols])
full_scaled = StandardScaler().fit(df[sensor_cols]).transform(df[sensor_cols])

def make_sequences(arr, seq_len):
    return np.array([arr[i:i+seq_len] for i in range(len(arr) - seq_len)])

train_seqs = make_sequences(normal_scaled, seq_len)
all_seqs = make_sequences(full_scaled, seq_len)

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden=32, latent=8):
        super().__init__()
        self.enc = nn.LSTM(n_features, hidden, batch_first=True)
        self.enc2 = nn.LSTM(hidden, latent, batch_first=True)
        self.dec = nn.LSTM(latent, hidden, batch_first=True)
        self.dec2 = nn.LSTM(hidden, n_features, batch_first=True)

    def forward(self, x):
        out, _ = self.enc(x)
        out, (h, _) = self.enc2(out)
        latent_rep = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        out, _ = self.dec(latent_rep)
        out, _ = self.dec2(out)
        return out

# training loop, batched inference, and thresholding: see notebook
```

### 4. Evaluation

For each known failure, check whether an anomaly was flagged in the N hours beforehand
(lead time), and measure the false-positive rate during confirmed `NORMAL` operation.

```python
def evaluate(results, score_col, anomaly_col, failure_times, lookback_hours):
    lead_times = []
    for t in failure_times:
        window_before = results[(results["timestamp"] >= t - pd.Timedelta(hours=lookback_hours)) &
                                 (results["timestamp"] <= t)]
        flagged = window_before[window_before[anomaly_col]]
        lead_times.append((t - flagged["timestamp"].min()).total_seconds() / 3600 if len(flagged) else 0)

    fpr = results.loc[results["machine_status"] == "NORMAL", anomaly_col].mean()
    caught = sum(l > 0 for l in lead_times)
    return {
        "failures_caught": f"{caught}/{len(failure_times)}",
        "avg_lead_time_hrs": np.mean([l for l in lead_times if l > 0]) if caught else 0,
        "false_positive_rate_on_normal": fpr,
    }
```

## Results

| Model | Failures caught | Avg. lead time | False positive rate (normal ops) |
|---|---|---|---|
| Isolation Forest | 1 / 7 | 47.4 hrs | 2.5% |
| LSTM Autoencoder | 3 / 7 | 13.7 hrs | 0.3% |

## Summary & Conclusion

The LSTM Autoencoder clearly outperformed the Isolation Forest baseline on this task,
catching three times as many real failures while producing an order of magnitude fewer
false positives during normal operation.

The Isolation Forest's single "catch" is not strong evidence of real detection: its
false-positive rate (2.5%) is high enough that, over a 48-hour evaluation window with
per-minute readings (~2,880 points), a stray flagged point landing near the start of the
window by chance is expected. Consistent with this, its reported lead time sat almost
exactly at the edge of whatever lookback window was used (6 hrs, then 48 hrs), in both
cases — a signature of the metric picking up baseline noise rather than a genuine
early-warning signal. In short, rolling mean/std features computed independently per
timestep don't give Isolation Forest enough structure to separate real pre-failure drift
from ordinary sensor noise.

The LSTM Autoencoder, by contrast, models entire 30-step sequences and is trained
exclusively on confirmed normal operation. Its reconstruction error rises specifically
when a sequence deviates from patterns the model has learned as "healthy" — capturing
the shape of degradation over time rather than judging each row in isolation. That
sequence-level view is what let it catch more failures with meaningfully less noise.

**Takeaway:** for this dataset, temporal/sequence modeling (LSTM Autoencoder) is a
substantively better fit than a per-row anomaly detector like Isolation Forest, even
with engineered rolling-window features. Isolation Forest remains useful as a fast,
cheap baseline to confirm a problem is learnable at all, but it is not the right final
model here.

### Possible next steps
- Lower the LSTM's anomaly threshold percentile to trade some false positives for a
  higher catch rate on the remaining 4 failures.
- Try longer input sequences (e.g. 60–120 steps) to give the LSTM more context per
  prediction.
- Engineer sensor-specific features (e.g. rate of change) as additional Isolation
  Forest inputs to see whether its ceiling is truly structural or just feature-limited.

## Dataset

- [Pump Sensor Data (Kaggle)](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) — 53 sensors, ~220,000 minute-level readings, 7 real recorded failures.
