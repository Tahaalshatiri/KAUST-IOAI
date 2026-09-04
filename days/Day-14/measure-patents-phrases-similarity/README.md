# Patent Phrase Similarity

Predict how similar two technical phrases are in a patent context.

Files:
- train.csv: id, anchor, target, context, score
- test.csv: id, anchor, target, context
- sample_submission.csv: id, score
- score_meanings.csv: score explanations
- context_summary.csv: context statistics

Metric: Pearson correlation.

Scores should be between 0 and 1.
