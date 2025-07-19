# Machin-learning-project-4-mnist-bayes-classifiers-dataml100-

## Task Description – Week 4: MNIST & Bayes Classifiers
This project focuses on implementing and comparing three different classifiers for the MNIST and Fashion MNIST datasets using Python:

1-Nearest Neighbor (1-NN)

Implemented using sklearn.neighbors.KNeighborsClassifier.

Images are flattened to 784-dimensional vectors.

Classifier is trained and tested, and accuracy is computed using a custom function.

Naive Bayes Classifier

Assumes pixel-wise independence (diagonal covariance).

Computes mean and variance for each pixel/class.

Uses log-likelihood to avoid numerical underflow.

Adds small variance or Gaussian noise to improve stability.

Full Multivariate Bayes Classifier

Uses full 784×784 covariance matrices per class.

Calculates log-likelihoods with scipy.stats.multivariate_normal.

Tests whether noise helps with singularity or low-rank matrices.

Each classifier is evaluated on both the original MNIST and Fashion MNIST datasets, and classification accuracy is reported.

