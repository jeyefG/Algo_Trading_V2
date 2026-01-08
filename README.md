# State Engine Training (Console UI)

## Uso rápido

```bash
python scripts/train_state_engine.py \
  --symbol XAUUSD \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --model-out models/xauusd_state_engine.pkl \
  --report-out reports/xauusd_state_engine.json
```

Opciones útiles:

- `--log-level INFO|DEBUG|WARNING`
- `--min-samples 2000`
- `--split-ratio 0.8`
- `--no-rich` (salida básica sin Rich)

## Before (salida anterior, simplificada)

```
Training complete.
            ALLOW_trend_pullback  ALLOW_balance_fade  ALLOW_transition_failure
2024-05-01                    0                  1                        0
2024-05-02                    0                  0                        0
...
```

## After (nueva UI de consola)

```
stage=descarga_h1
download_rows=22080 elapsed=3.14s
stage=build_dataset
features_raw=22080 labels_raw=22080 elapsed=0.82s
stage=align_and_clean
aligned_samples=20350 dropped_nan=1730 elapsed=0.04s
stage=split
n_train=16280 n_test=4070 split_ratio=0.80 elapsed=0.00s
stage=train_model
train_elapsed=2.10s
stage=evaluate
accuracy=0.6243 f1_macro=0.5981 elapsed=0.01s
stage=predict_outputs
outputs_rows=20350 elapsed=0.05s
stage=gating
gating_allow_rate=34.20% elapsed=0.01s
stage=save_model
model_path=models/xauusd_state_engine.pkl elapsed=0.00s

=== State Engine Training Summary ===
Symbol: XAUUSD
Period: 2024-01-01 -> 2025-12-31
Samples: 20350 (train=16280, test=4070)
Baseline: BALANCE (45.10%)
Accuracy: 0.6243 | F1 Macro: 0.5981
Gating allow rate: 34.20%
Model saved: models/xauusd_state_engine.pkl
Report saved: reports/xauusd_state_engine.json
```
