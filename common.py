import numpy as np

def add_epsilon_to_matrix(M):
  ''' Ensure numerical stability by adding epsilon to
      each entry and normalising each row to sum to
      one.
  '''
  EPSILON = 1.0e-10
  nrows = M.shape[0]
  for r in range(nrows):
    M[r] = add_epsilon(M[r], EPSILON)
  print(M)
  return M

def add_epsilon(r, eps):
  ''' Add eps to each entry and then normalise
      row sum to one.
  '''
  n = len(r)
  new_r = [(el+eps)/(1+eps*n)  for el in r]
  print(f'[add_epsilon] sum: {np.sum(new_r)}')
  return new_r
