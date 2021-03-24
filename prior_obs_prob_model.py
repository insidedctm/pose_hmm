import argparse
import glob
import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal

class PriorObservationProbModel:

  def __init__(self, model_path):
    self.pca, self.mu, self.sigma = input_params(model_path)

  def __call__(self, pose_landmarks, frame_height, frame_width):
    assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
    x = pose_landmarks.reshape(1,99) 
    reduced_x = self.pca.transform(x)
    print(f'[DEBUG] {multivariate_normal.pdf(self.mu, mean=self.mu, cov=self.sigma)}')
    return multivariate_normal.pdf(reduced_x, mean=self.mu, cov=self.sigma)   

def prior_obs_prob_model(input_path, output_path):
  print(f'input: {input_path}; output: {output_path}')

  pose_data = get_pose_data(input_path)

  reduced_pose_data_2d, _ = run_pca(pose_data, n_components=2)
  import matplotlib.pyplot as plt
  plt.scatter(reduced_pose_data_2d[:,0], reduced_pose_data_2d[:,1])
  plt.show()

  reduced_pose_data, pca = run_pca(pose_data, n_components=5)

  mu, sigma = get_mvn_params(reduced_pose_data)

  output_params(pca, mu, sigma, output_path)

def get_pose_data(path):
  print(f'(get_pose_data) path={path}')
  poses = []
  video_paths = glob.glob(f'{path}/*')
  for video in video_paths:
    print(f'loading {video}')    
    cap = cv2.VideoCapture(video)
    # initialise Pose estimator for whole video
    pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    )
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      results = pose.process(frame)  
      pose_landmarks = results.pose_landmarks
      # convert landmarks to list
      if pose_landmarks is not None:
         # Get landmarks.
         frame_height, frame_width = frame.shape[0], frame.shape[1]
         pose_landmarks = np.array(
              [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                 for lmk in pose_landmarks.landmark],
                dtype=np.float32)
         assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
         poses.append(pose_landmarks.reshape(99))
    cap.release()
  poses = np.array(poses)
  return poses

def run_pca(data, n_components=10):
  print('(run_pca)')
  pca = PCA(n_components=n_components)
  pca.fit(data) 
  print(f'(run_pca) transformed data shape: {pca.transform(data).shape}')
  return pca.transform(data), pca

def get_mvn_params(data):
  print('(get_mvn_params)')
  mu = data.mean(axis=0)
  print(mu)
  print(mu.shape)
  cov = np.cov(data.T)
  print(cov)
  print(cov.shape)
  return mu, cov

def output_params(pca, mu, sigma, path):
  print('(output_params)')
  import pickle as pk
  pk.dump(pca, open(f'{path}/pca.pkl','wb'))
  np.savetxt(f'{path}/mu.csv'   , mu   , delimiter=",")
  np.savetxt(f'{path}/sigma.csv', sigma, delimiter=",")

def input_params(path):
   print('(input_params)')
   import pickle as pk
   pca = pk.load(open(f'{path}/pca.pkl','rb'))
   mu = np.loadtxt(f'{path}/mu.csv', delimiter=",")
   sigma = np.loadtxt(f'{path}/sigma.csv', delimiter=",")
   return pca, mu, sigma 

def parse_args():
  parser = argparse.ArgumentParser(description='Learn prior observation model parameters')
  parser.add_argument('inp', help='Directory containing source videos of people performing an exercise')
  parser.add_argument('out', help='Directory to output parameters for prior observation probability model')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()

  prior_obs_prob_model(args.inp, args.out)
