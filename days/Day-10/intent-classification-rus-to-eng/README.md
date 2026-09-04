# Cross-Lingual Russian Intent Detection and Slot Filling

You are given labeled English NLU data and must predict intents and slot tags for Russian utterances.

## Files

- `train.conll` — English labeled training data.
- `test.conll` — Russian unlabeled test data for leaderboard submission.
- `sample_submission.csv` — required submission format.
- `intent_labels.csv` — possible intent labels.
- `slot_labels.csv` — possible BIO slot tags.
- `ru_en_pairs.jsonl` — optional Russian-English sentence pairs, if present in the source dataset.
- `unlabeled_texts.txt` — optional Russian unlabeled text, if present in the source dataset.

## Submission Format

Submit a CSV with columns:

```text
id,intent,slots
```

`slots` must contain one whitespace-separated BIO tag per token in the corresponding `test.conll` sample.

Example:

```csv
id,intent,slots
test_00000,GetWeather,O O B-city O
test_00001,BookRestaurant,O B-restaurant_name O O
```

## Metric

The score is the average of two weighted F1 scores:

```text
score = 0.5 * weighted_f1(intent) + 0.5 * weighted_f1(slot_tags)
```

Slot tags are flattened across all test tokens before scoring.
