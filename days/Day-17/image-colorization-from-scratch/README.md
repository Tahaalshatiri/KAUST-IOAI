# CIFAR-10 Image Colorization

Given a grayscale CIFAR-10 image, predict its original RGB color image.

Files:
- train.csv: image ids and paths to grayscale/color training images
- test.csv: image ids and paths to grayscale test images
- sample_submission.csv: required submission format
- train_gray/: grayscale training images
- train_color/: original color training images
- test_gray/: grayscale test images

Submission format:
`row_id,value` where `row_id` is `imageid_y_x_channel`, and value is a number from 0 to 255.

No pretrained models are allowed.

## Baseline

`baseline.ipynb` — from-scratch CNN predicting a color delta over the gray image. Val MAE **12.2**/255 (beats gray-replication 12.9).
