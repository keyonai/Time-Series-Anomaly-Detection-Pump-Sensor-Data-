# Time-Series Anomaly Detection (Pump Sensor Data)

Unsupervised anomaly detection on industrial pump sensor data — flags abnormal sensor behavior ahead of real pump failures.

## Models

- Isolation Forest (rolling mean/std features) — baseline
- LSTM Autoencoder (PyTorch) — trained on normal operation, flags high reconstruction error as anomalous

## Run it

```bash
pip install kagglehub scikit-learn torch pandas numpy matplotlib
```

Open `Time_Series_Anomaly_Detection_(Pump_Sensor_Data).ipynb` in Google Colab. Dataset: [Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) (53 sensors, ~220,000 readings, 7 recorded failures), pulled via `kagglehub`.

## Results

| Model | Failures caught | Avg. lead time | False positive rate |
|---|---|---|---|
| Isolation Forest | 1 / 7 | 47.4 hrs | 2.5% |
| LSTM Autoencoder | 3 / 7 | 13.7 hrs | 0.3% |

## Summary

The LSTM Autoencoder beat Isolation Forest, catching 3x more failures with far fewer false positives. It models whole sequences instead of single rows, so it picks up the shape of degradation over time rather than judging each timestep on its own — that's what gave it the edge.

## Next steps

- Lower the LSTM's anomaly threshold to catch more of the remaining failures
- Try longer input sequences for more context per prediction
- Add rate-of-change features to see if Isolation Forest's ceiling is fixable
