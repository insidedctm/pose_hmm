import numpy as np

# Generic 5 state transition matrix
# Setup transition matrix (Pair potentials)
P = np.array([
      [ 0.46, 0.44, 0.001, 0.099, 0.0   ],
      [ 0.0,  0.7,  0.2,   0.099, 0.001 ],
      [ 0.0,  0.001,  0.7,   0.298, 0.001 ],
      [ 0.0,  0.0005,  0.0015,   0.898, 0.1],
      [ 0.0,  0.0,  0.0,   0.0,   1.0]
])
P = np.savetxt('transition_matrix.csv', P, delimiter=',')


# Generic 4 state transition matrix
# Setup transition matrix (Pair potentials)
P = np.array([
      [ 0.46, 0.44,  0.1,    0.0  ],
      [ 0.0,  0.78,  0.18,   0.04 ],
      [ 0.0,  0.18,  0.78,   0.04 ],
      [ 0.01, 0.01,  0.01,   0.97 ]
])
P = np.savetxt('transition_matrix_4_states.csv', P, delimiter=',')


