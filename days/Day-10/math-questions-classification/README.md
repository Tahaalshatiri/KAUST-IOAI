# Math Topic Classification

Classify each math problem into one of 8 topics.

Labels:
0 Algebra
1 Geometry and Trigonometry
2 Calculus and Analysis
3 Probability and Statistics
4 Number Theory
5 Combinatorics and Discrete Math
6 Linear Algebra
7 Abstract Algebra and Topology

Files:
- train.csv: id,Question,label
- test.csv: id,Question
- sample_submission.csv: id,label
- labels.csv

Metric: Macro F1. Rare classes matter.

Suggested rule: no transformers / no LLM embeddings.
Recommended methods: TF-IDF word/character n-grams, Logistic Regression, Linear SVM, Naive Bayes.
