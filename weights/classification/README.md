# Trained Traffic Sign Classifiers

This directory contains weights for traffic sign classifers, trained on ATSD. There are different
versions, each consisting of an HDF5 file and a JSON file. The HDF5 files contain the weights in the usual
Keras/Tensorflow format, and the JSON files contain information about input normalization and list of class-IDs.

Models were trained on augmented data and/or traffic sign images (with background context) extracted from ATSD-Scenes,
as described in `Classification_Preparation.ipynb`.

## v7

Trained on the training set of ATSD-Signs with geometric- and LED augmentation.
The performance of the model on independent test sets is as follows:
* Public test set: 97.77% accuracy, 95.75% balanced accuracy
* Internal test set: 97.33% accuracy, 94.95% balanced accuracy

## v10

Trained on all publicly available data of ATSD-Signs with geometric- and LED augmentation.
The performance of the model on independent test sets is as follows:
* Internal test set: 98.53% accuracy, 97.90% balanced accuracy