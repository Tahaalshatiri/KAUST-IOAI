# Pet Image Classification

Classify each pet image into one of the anonymous classes.

## Files

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `labels.csv`
- `dataset_stats.csv`
- `train_images/`
- `test_images/`

## Columns

`train.csv`:

```text
id,image_path,label
```

`test.csv`:

```text
id,image_path
```

`sample_submission.csv`:

```text
id,label
```

## Metric

Accuracy.

## Note

The test images may not have exactly the same visual distribution as the training images. Robust preprocessing and validation are important.
