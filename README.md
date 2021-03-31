# Hidden Markov Model Pose-Exercise Model

## Overview

Represent a sequence of poses from an exercise as a HMM. The position in the HMM can be read from the decoded states using the Viterbi algorithm (see
Machine Vision, Prince s11.2 for details and notation)

## Model

The state transition model is hand-coded to represent prior beliefs on how a user progresses through an exercise. This gives us the pair potentials
P(w_t-1, w_t).

The observation model requires unary potentials U(w_t) p.t. P(x_t|w_t) = P(w_t|x_t)P(x_t)/P(w_t). Since we only need the probablity as a function 
of w_t we treat P(x_t) as a constant. P(w_t) is the prior probability distribution at time t, in the absence of firm belief otherwise we treat as uniform and 
for the purposes of finding the most probable state can be treated as a constant. Therefore U(w_t) = P(w_t|x_t); P(w_t|x_t) is given by the KNN Pose 
Classifier output. 

## Running the demo
The demos assume that `PYTHONPATH` includes the directory of the `pose_knn_classifier`.

To run the online version of the viterbi algorithm on a pre-recorded video
```{bash}
python demo_hmm_segmenter.py <source-video> <classifier-samples-folder> --mode Online
```

e.g. 
```{bash}
PYTHONPATH=../../pose_knn_classifier/ python demo_hmm_segmenter.py Robin\ childs\ pose\ test.mp4 samples --mode Online
```

To save the output to a video file use `--mode Offline`. The file is saved in the current directory as `output.avi`.

### Pose Classifier configuration
The pose classifier needs to provide a classification for each state in the HMM, given a pose. The classifier is configured via a set of CSVs contained
in a `samples` folder. The folder should contain 1 CSV for each state, with each file being named according to the state it represents. 

For example the `childs pose` model has the following structure 
```{bash}
childs_pose_allfours.csv
childs_pose_backonheels.csv
childs_pose_end.csv
childs_pose_start.csv
childs_pose_throughshoulders.csv
```
