# Hidden Markov Model Pose-Exercise Model

## Overview

Represent a sequence of poses from an exercise as a HMM. The position in the HMM can be read from the decoded states using Viterbi algorithm.

## Training Pipeline

The state transition model is hand-coded to represent prior beliefs on how a user progresses through and exercise

The observation model requires P(x_t|w_t) p.t. P(w_t|x_t)P(x_t). P(w_t|x_t) is given by the KNN Pose Classifier output.
P(x_t) can be estimated as MVN(mu, sigma) from the all the poses used in the training videos for a given exercise.

### Estimating P(x_t)

Since x_t is 99-dimensional first reduce dimensions using PCA retaining, say, the first 10 components. Using the dimension reduced dataset
calculate mean and covariance matrix for the dataset, using these as the parameters for MVN. Output is then the 99x10 PCA transformation 
matrix and the 10x1 mean and 10x10 covariance matrix.

The input to the estimation process is a folder containing 1 or more videos of a user performing the target exercise. The process first
extracts frames from all the videos and retrieves poses using BlazePose. Then we run PCA on the pose dataset and extract the dimension-reduced
dataset. Finally we calculate the mean and covariance on the dimension-reduced dataset. The PCA transformation matrix, mean vector and 
covariance matrix are output to the target directory.

```{bash}
python prior_obs_prob_model.py input_dir output_dir
```
