# Time-Series Anomaly Detection (Pump Sensor Data)

Unsupervised anomaly detection on industrial pump sensor data. Detects abnormal sensor behavior that precedes real pump failures, without being told in advance where the failures are.

## Models

- Isolation Forest (rolling mean/std features) — baseline
- LSTM Autoencoder (PyTorch) — trained on normal operation, flags high reconstruction error as anomalous

## Run it

```bash
pip install kagglehub scikit-learn torch pandas numpy matplotlib
```

Open `Time_Series_Anomaly_Detection_(Pump_Sensor_Data).ipynb` in Google Colab. Uses the [Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) dataset (53 sensors, ~220,000 readings, 7 recorded failures), pulled automatically via `kagglehub`.

## Results

| Model | Failures caught | Avg. lead time | False positive rate |
|---|---|---|---|
| Isolation Forest | 1 / 7 | 47.4 hrs | 2.5% |
| LSTM Autoencoder | 3 / 7 | 13.7 hrs | 0.3% |

## Summary

The LSTM Autoencoder outperformed Isolation Forest, catching 3x more failures with far fewer false positives. Isolation Forest's one "catch" isn't strong evidence of real detection — its false-positive rate is high enough that a stray flagged point landing near the start of the lookback window is expected by chance, and its lead time consistently sat right at the window edge rather than showing genuine early warning.

The LSTM's advantage comes from modeling whole sequences instead of individual rows: it learns what a normal 30-step pattern looks like and flags sequences that deviate from it, capturing the shape of degradation over time rather than judging each timestep in isolation. That's what let it catch more failures with a cleaner signal.

**Takeaway:** sequence modeling is a meaningfully better fit for this problem than a per-row detector, even with rolling-window features. Isolation Forest is still useful as a fast baseline to confirm the problem is learnable, but it isn't the right final model here.
