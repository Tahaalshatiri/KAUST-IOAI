# Coin Value Counting

Simplified task based on the Polish "Maszynka do Liczenia Monet" coin-detection notebook.

## Goal

Given an image of Polish coins, predict the total value of all visible coins in **grosz**.

Examples:

- 1 zloty = 100 grosz
- 2 zloty = 200 grosz
- 5 zloty = 500 grosz

## Files

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `coin_values.csv`
- `split_info.csv`
- `train_images/`
- `test_images/`

## Train columns

```csv
id,image_path,total_value_grosz
```

## Test columns

```csv
id,image_path
```

## Submission columns

```csv
id,total_value_grosz
```

## Metric

The hidden metric is inverse mean absolute error:

```text
score = 1 / (1 + MAE_in_zloty)
```

Higher is better. Perfect prediction gives 1.0.

## Fixed split

`SPLIT_SEED = 2026`

## Baseline

`baseline.ipynb` — frozen ResNet-18 features + Ridge regression. CV MAE **322 grosz** (mean target 711). See notebook for the intended Hough-circles approach.
