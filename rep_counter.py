from math import floor, ceil

class RepCounter:
  def __init__(self, rep_start_state, rep_end_state):
    print(f'Creating RepCounter with start={rep_start_state}, end={rep_end_state}')
    self.start = rep_start_state
    self.end   = rep_end_state
    self.n_reps = -0.5 # we add 0.5 on the first entry to the start state
    self.looking_for_state = self.start

  def __call__(self, current):
    if current == self.end:
      if self.looking_for_state == self.end:
        self.n_reps += 0.5
        self.looking_for_state = self.start

    if current == self.start:
      if self.looking_for_state == self.start:
        self.n_reps += 0.5
        self.looking_for_state = self.end

    return self.rep_count()

  def rep_count(self):
    if self.n_reps > 0:
      return floor(self.n_reps)
    else:
      return ceil(self.n_reps)

  @staticmethod
  def reps_from_states(states, rep_start_state, rep_end_state):
    ''' Takes a list of states, constructs a RepCounter and 
        submits each state in the list to the counter, returning
        the list of reps returned for each state
    '''
    counter = RepCounter(rep_start_state, rep_end_state)
    reps = [counter(state) for state in states]
    return reps

if __name__ == '__main__':
  example_states = [0,0,0,1,1,2,3,2,2,1,2,1,2,2,3,2,1,4]
  expected       = [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2]
  counts = RepCounter.reps_from_states(example_states, 1, 3) 
  print(f'expected: {expected}')
  print(f'got:      {counts}')


  example_states = [0,0,0,1,1,1,1,2,1,1,1,2,2,2,2,1,1,1,3]
  expected       = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2]
  counts = RepCounter.reps_from_states(example_states, 1, 2)
  print(f'expected: {expected}')
  print(f'got:      {counts}')


