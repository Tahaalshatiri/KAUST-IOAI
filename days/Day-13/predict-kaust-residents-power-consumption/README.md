# Mini Enefit: Solar Prosumer Energy Forecasting

Predict hourly electricity consumption and production for Estonian prosumer groups.

Each row describes one hour for a segment defined by county, business/customer status, contract/product type, and whether the row is consumption or production.

## Files

- `train.csv` — training rows with target.
- `test.csv` — hidden test rows without target.
- `sample_submission.csv` — required submission format.
- `dataset_stats.csv` — basic dataset information.
- `feature_guide.csv` — short feature explanations.

## Task

Predict the numeric `target` for every row in `test.csv`.

## Submission

Submit a CSV with:

```csv
id,target
test_000000,0.0
test_000001,0.0
```

## Metric

The competition uses:

```text
score = 1 / (1 + MAE)
```

Higher is better. Perfect score is 1.0.

## Tips

Useful features usually include:

- `lag_168` — target from the same hour one week ago
- hour, day of week, month
- `is_consumption`
- installed solar capacity
- solar radiation and cloud cover
- mean target features by prediction unit

Production rows are strongly connected to solar radiation and installed capacity.
Consumption rows are often more connected to hour-of-day and weekday patterns.
