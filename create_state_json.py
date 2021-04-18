import json

def write_to_file(filename, state_names, state_descriptions):
  states = {'state_names': state_names, 'state_descriptions': state_descriptions}

  # open output file for writing
  with open(filename, 'w') as filehandle:
      json.dump(states, filehandle)


###################################
#  Y Squats
###################################
state_names = ['y_squats_begin', 'y_squats_up', 'y_squats_down', 'y_squats_finish']
state_descriptions = [
	'Waiting to Start',
	'Hands Up',
	'Squat',
	'Finished'
]

write_to_file('states_y_squats.json', state_names, state_descriptions)

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


###################################
#  Kneeling pushup shoulder tap
###################################
state_names = ['kneeling_pushups_shoulder_tap_begin', 'kneeling_pushups_shoulder_tap_down', 'kneeling_pushups_shoulder_tap_up', 'kneeling_pushups_shoulder_tap_ltap', 'kneeling_pushups_shoulder_tap_rtap', 'kneeling_pushups_shoulder_tap_finish']
state_descriptions = [
	'Waiting to Start',
	'Down',
	'Up',
        'Left Shoulder Tap',
        'Right Shoulder Tap',
	'Finished'
]

write_to_file('states_kneeling_pushups_shoulder_tap.json', state_names, state_descriptions)


###################################
#  Reverse Lunge Hands Over Head
###################################
state_names = ['reverse_lunge_hands_over_head_begin', 'reverse_lunge_hands_over_head_down', 'reverse_lunge_hands_over_head_up', 'reverse_lunge_hands_over_head_finish']
state_descriptions = [
	'Waiting to Start',
	'Down',
	'Up',
	'Finished'
]

write_to_file('states_reverse_lunge_hands_over_head.json', state_names, state_descriptions)


###################################
#  Diamond Situps
###################################
state_names = ['diamond_situps_begin', 'diamond_situps_lying', 'diamond_situps_upright', 'diamond_situps_fold', 'diamond_situps_finish']
state_descriptions = [
	'Waiting to Start',
	'Lying back',
	'Upright',
        'Forward',
	'Finished'
]

write_to_file('states_diamond_situps.json', state_names, state_descriptions)


###################################
#  Push Press
###################################
state_names = ['push_press_begin', 'push_press_up', 'push_press_down', 'push_press_finish']
state_descriptions = [
	'Waiting to Start',
	'Up position',
	'Down position',
	'Finished'
]

write_to_file('states_push_press.json', state_names, state_descriptions)


###################################
#  Goblet Squat
###################################
state_names = ['goblet_squat_begin', 'goblet_squat_up', 'goblet_squat_down', 'goblet_squat_finish']
state_descriptions = [
	'Waiting to Start',
	'Up position',
	'Down position',
	'Finished'
]

write_to_file('states_goblet_squat.json', state_names, state_descriptions)


###################################
#  Bent Over Row
###################################
state_names = ['bent_over_row_begin', 'bent_over_row_up', 'bent_over_row_pushdown', 'bent_over_row_down', 'bent_over_row_pullup', 'bent_over_row_finish']
state_descriptions = [
	'Waiting to Start',
	'Up position',
        'Pushing down',
	'Down position',
        'Pulling up',
	'Finished'
]

write_to_file('states_bent_over_row.json', state_names, state_descriptions)


###################################
#  Forward Fold
###################################
state_names = ['forward_fold_begin', 'forward_fold_up', 'forward_fold_bending', 'forward_fold_down', 'forward_fold_finish']
state_descriptions = [
	'Waiting to Start',
	'Up position',
        'Bending down',
	'Down position',
	'Finished'
]

write_to_file('states_forward_fold.json', state_names, state_descriptions)


###################################
#  Down Dog
###################################
state_names = ['down_dog_begin', 'down_dog_standing', 'down_dog_allfours', 'down_dog_hipsup', 'down_dog_extended', 'down_dog_finish']
state_descriptions = [
	'Waiting to Start',
	'Standing',
        'All fours',
	'Hips up',
        'Fully extended',
	'Finished'
]

write_to_file('states_down_dog.json', state_names, state_descriptions)


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


