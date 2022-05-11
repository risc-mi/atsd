# Results of Detection+Classification Pipelines

This directory contains results of detection+classification pipelines, trained and evaluated on ATSD. There are
different versions, each consisting of a single CSV file.

Results can be evaluated following the approach outlined in `Evaluation.ipynb`.

## 1_7

Trained on the training set of ATSD. The classifier was trained with geometric- and LED augmentation, and actually
corresponds to version v7 in the `/weights` folder.
Evaluation results are as follows:
* Detection only: 87.65% mAP
* Detection+classification: 89.66% mAP