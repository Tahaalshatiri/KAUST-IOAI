# Tricy Table Data

You are given a table with missing values and hidden patterns.

Your task:

```text
features -> target
```

## Files

- `train_tables.csv`
- `test_tables.csv`
- `sample_submission.csv`
- `feature_summary.csv`

## Submission format

```csv
id,target
test_00000,123.45
test_00001,67.89
```

## Metric

Custom score:

```text
score = 1 / (1 + MAE)
```

Larger is better. Perfect score is 1.0.

## Notes

The dataset intentionally contains missing values. Good strategies:

- study correlations
- impute missing values carefully
- use the time columns `day`, `hour`, `minute`
- try target transforms such as log target
- train with LightGBM / tree models
