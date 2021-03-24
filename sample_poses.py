import argparse
import glob
import pandas as pd

def sample_poses(samples_path, num_samples, class_name, target_path):
  csvs = glob.glob(f'{samples_path}/*.csv')
  dfs = [pd.read_csv(csv, header=None) for csv in csvs]
  poses = pd.concat(dfs, ignore_index=True)
  print(poses)

  sample = poses.sample( n = num_samples )
  print(sample)

  output_path = f'{target_path}/{class_name}.csv'
  sample.to_csv(output_path, index=None, columns=None)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('samples_path', help='folder containing class samples csvs')
  parser.add_argument('num_samples', type=int, help='number of samples to extract')
  parser.add_argument('class_name', help='name to give to the samples - stored in target_path as class_name.csv')
  parser.add_argument('target_path', help='path to store the samples csv in')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()

  sample_poses(args.samples_path, args.num_samples, args.class_name, args.target_path)
