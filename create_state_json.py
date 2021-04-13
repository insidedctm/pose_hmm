import json

def write_to_file(filename, state_names, state_descriptions):
  states = {'state_names': state_names, 'state_descriptions': state_descriptions}

  # open output file for writing
  with open(filename, 'w') as filehandle:
      json.dump(states, filehandle)

###################################
#  Childs Pose
###################################
state_names = ['childs_pose_start', 'childs_pose_allfours', 'childs_pose_backonheels', 'childs_pose_throughshoulders', 'childs_pose_end']
state_descriptions = [
     'Waiting to Start',
     'On all fours',
     'Back on heels',
     'Full extension',
     'Finished' ]

write_to_file('states_childs_pose.json', state_names, state_descriptions)

###################################
#  Crawlouts
###################################
state_names = ['crawlouts_begin', 'crawlouts_standing', 'crawlouts_crawling', 'crawlouts_plank', 'crawlouts_finish']
state_descriptions = [
	'Waiting to Start',
	'Standing',
	'Crawling',
	'Plank',
	'Finished'
]

write_to_file('states_crawlouts.json', state_names, state_descriptions)

###################################
#  Plank to T
###################################
state_names = ['plank_to_t_begin', 'plank_to_t_plank', 'plank_to_t_rotate', 'plank_to_t_tposition', 'plank_to_t_finish']
state_descriptions = [
	'Waiting to Start',
	'Plank',
	'Rotating',
	'T position',
	'Finished'
]

write_to_file('states_plank_to_t.json', state_names, state_descriptions)


###################################
#  3 Step Drill
###################################
state_names = ['3_step_drill_begin', '3_step_drill_left', '3_step_drill_step', '3_step_drill_right', '3_step_drill_finish']
state_descriptions = [
	'Waiting to Start',
	'Left Hold',
	'Step',
	'Right Hold',
	'Finished'
]

write_to_file('states_3_step_drill.json', state_names, state_descriptions)


###################################
#  Squat to Plank
###################################
state_names = ['squat_to_plank_begin', 'squat_to_plank_stand', 'squat_to_plank_crouch', 'squat_to_plank_plank', 'squat_to_plank_finish']
state_descriptions = [
	'Waiting to Start',
	'Stand',
	'Squat',
	'Plank',
	'Finished'
]

write_to_file('states_squat_to_plank.json', state_names, state_descriptions)


###################################
#  Predator Jacks
###################################
state_names = ['predator_jacks_begin', 'predator_jacks_up', 'predator_jacks_down', 'predator_jacks_finish']
state_descriptions = [
	'Waiting to Start',
	'Arms in',
	'Arms out',
	'Finished'
]

write_to_file('states_predator_jacks.json', state_names, state_descriptions)


