import numpy as np
from common import add_epsilon_to_matrix

# Generic 5 state transition matrix
# Setup transition matrix (Pair potentials)
P = np.array([
      [ 0.46, 0.44, 0.001, 0.099, 0.0   ],
      [ 0.0,  0.7,  0.2,   0.099, 0.001 ],
      [ 0.0,  0.001,  0.7,   0.298, 0.001 ],
      [ 0.0,  0.0005,  0.0015,   0.898, 0.1],
      [ 0.0,  0.0,  0.0,   0.0,   1.0]
])

P = add_epsilon_to_matrix(P)
P = np.savetxt('transition_matrix.csv', P, delimiter=',')


# Generic 4 state transition matrix
# Setup transition matrix (Pair potentials)
P = np.array([
      [ 0.46, 0.44,  0.1,    0.0  ],
      [ 0.0,  0.78,  0.18,   0.04 ],
      [ 0.0,  0.18,  0.78,   0.04 ],
      [ 0.01, 0.01,  0.01,   0.97 ]
])

P = add_epsilon_to_matrix(P)
P = np.savetxt('transition_matrix_4_states.csv', P, delimiter=',')


# Generic 6 state transition matrix
# Setup transition matrix (Pair potentials)
P = np.array([
      [ 0.44, 0.19, 0.19, 0.18, 0.0, 0.0   ],
      [ 0.0,  0.7,  0.28,   0.005, 0.005, 0.01 ],
      [ 0.0,  0.1,  0.7,   0.1, 0.1, 0. ],
      [ 0.0,  0.05,  0.25,   0.7, 0.0, 0.],
      [ 0.0,  0.05,  0.25,    0.0, 0.7, 0.],
      [ 0.0,  0.0,  0.0,   0.0, 0.0,  1.0]
])

P = add_epsilon_to_matrix(P)
P = np.savetxt('transition_matrix_6_states.csv', P, delimiter=',')


