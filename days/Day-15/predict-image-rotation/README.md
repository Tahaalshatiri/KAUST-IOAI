# Pet Rotation Prediction

Predict the rotation label of a circle-cropped pet image.

Labels:
1=0°, 2=90°, 3=180°, 4=270°.

Files:
- train.csv
- test.csv
- sample_submission.csv
- rotation_labels.csv
- dataset_stats.csv
- train_images/
- test_images/

Metric:
score = 1 / (1 + circular_mse)

Because rotation is circular, label 1 and label 4 are adjacent.
