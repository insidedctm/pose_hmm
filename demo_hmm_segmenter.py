import argparse
from prior_obs_prob_model import PriorObservationProbModel
from pose_embedder import FullBodyPoseEmbedder
from pose_classifier import PoseClassifier
import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
import numpy as np

MAX_FLOAT = np.finfo(np.float32).max

def demo_hmm_segmenter(video_path, classifier_samples_folder, prior_model_path):

  # setup a list to hold the frame number for frames that have a valid pose
  #   (required to align hmm state history when overlaying onto video)
  pose_frames = []

  prior_model = PriorObservationProbModel(prior_model_path)

  # Transforms pose landmarks into embedding.
  pose_embedder = FullBodyPoseEmbedder()

  # Classifies give pose against database of poses.
  pose_classifier = PoseClassifier(
      pose_samples_folder=classifier_samples_folder,
      pose_embedder=pose_embedder,
      top_n_by_max_distance=30,
      top_n_by_mean_distance=10)

  cap = cv2.VideoCapture(video_path)  
  # initialise Pose estimator for whole video
  pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
  )

  # Setup transition matrix (Pair potentials)
  P = np.array([
        [ 0.46, 0.44, 0.001, 0.099, 0.0   ],
        [ 0.0,  0.7,  0.2,   0.099, 0.001 ],
        [ 0.0,  0.001,  0.7,   0.298, 0.001 ],
        [ 0.0,  0.0005,  0.0015,   0.898, 0.1],
        [ 0.0,  0.0,  0.0,   0.0,   1.0]
  ])

  P = make_array_neg_log(P)

  # Unary potentials for observations lattice (see Computer Vision (Prince) s11.2)
  U = [[1., 0., 0., 0., 0.]]

  cnt=0
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    results = pose.process(frame)
    pose_landmarks = results.pose_landmarks    
    if pose_landmarks:

      # add to the frame number list so we can align state with output frame when 
      # overlaying onto output video
      pose_frames.append(cnt)

      frame_height, frame_width = frame.shape[0], frame.shape[1]
      pose_landmarks = np.array(
              [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                 for lmk in pose_landmarks.landmark],
                dtype=np.float32)
      assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
      p_w_bar_x = {k:v/10. for k,v in sorted(pose_classifier(pose_landmarks).items(), key=lambda item: item[1], reverse=True)}
      print(f'P(w|x): {p_w_bar_x}')
  
      h, w = frame.shape[0], frame.shape[1]
      p_x = prior_model(pose_landmarks, h, w)
      print(f'[{cnt}] P(x): {p_x} (log={np.log(p_x)})')
  
      # add each p(w|x) to lattice
      U.append([
		p_w_bar_x['childs_pose_start'] if 'childs_pose_start' in p_w_bar_x else 0., 
		p_w_bar_x['childs_pose_allfours'] if 'childs_pose_allfours' in p_w_bar_x else 0., 
		p_w_bar_x['childs_pose_backonheels'] if 'childs_pose_backonheels' in p_w_bar_x else 0., 
		p_w_bar_x['childs_pose_throughshoulders'] if 'childs_pose_throughshoulders' in p_w_bar_x else 0., 
		p_w_bar_x['childs_pose_end'] if 'childs_pose_end' in p_w_bar_x else 0.])
    else:
      pose_frames.append(-1)  
    cnt += 1

  # compute most likely state sequence
  U = np.array(U)
  U = make_array_neg_log(U)
  # Transpose so columns are 1..T time steps and rows are 1..K states
  U = U.T  
  states = viterbi(U, P)
  print(states)

  # overlay onto video
  video_overlay(video_path, pose_frames, states)  
    
def viterbi(U, P):
  print('>>> viterbi')
  print('======== U =======')
  print(U)
  print('======== P =======')
  print(P)
  print('==================')

  # state history
  history = []

  K = U.shape[0]
  N = U.shape[1]
  print(f'running viterbi for {N} timesteps, with state space size {K}')
  for k in range(K):
    print(f'U[:,0]={U[:,0]}')
    print(f'U[{k},0]={U[k,0]}')

  for n in range(1,N):
    print(f'Time step {n}')
    for k in range(K):
      costs = [U[j,n-1] + P[j,k] + U[k,n] for j in range(K)]
      print(f'[n={n}, k={k}] costs = {costs}')
      U[k,n] = min(costs)
    print(f'U[:,n]={U[:,n]}')
    min_ix = np.argmin(U[:,n])
    history.append(min_ix)
    print(f'index {min_ix} is minimum cost ({U[min_ix,n]})')

  return history

def make_array_neg_log(A):
  shape = A.shape
  l = A.reshape(-1)
  l = np.array([neg_log(el) for el in l])
  return l.reshape(shape)

def neg_log(x):
  if x == 0.:
    return MAX_FLOAT
  else:
    return -np.log(x)

STATE_NAMES = [
     'Waiting to Start',
     'On all fours',
     'Back on heels',
     'Full extension',
     'Finished' ]

def video_overlay(video_path, frames_with_pose, states):
  print(len(frames_with_pose))
  print(len(states))
  overlay_states = np.full(len(frames_with_pose), -1)
  mask = [el >= 0 for el in frames_with_pose]
  overlay_states[mask] = states
  print(f'Overlaying video with these states: {overlay_states}')
  cap = cv2.VideoCapture(video_path)
  cnt = 0
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    if overlay_states[cnt] >= 0:
      state = overlay_states[cnt]
      state_name = STATE_NAMES[state]
      frame = overlay(frame, state_name)
    cv2.imshow('hmm demo', frame)
    cv2.waitKey(25)
    cnt += 1

def overlay(image, state):
  # font
  font = cv2.FONT_HERSHEY_SIMPLEX
  
  # org
  org = (50, 50)
  
  # fontScale
  fontScale = 1
   
  # Blue color in BGR
  color = (255, 0, 0)
  
  # Line thickness of 2 px
  thickness = 2

  # Using cv2.putText() method
  image = cv2.putText(image, str(state), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)    
  return image

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('video_path')
  parser.add_argument('classifier_samples_path')
  parser.add_argument('prior_model_path')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  print(args)

  demo_hmm_segmenter(args.video_path, args.classifier_samples_path, args.prior_model_path)

