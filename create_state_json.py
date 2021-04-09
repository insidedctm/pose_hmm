import json

# define list with values
#state_names = ['childs_pose_start', 'childs_pose_allfours', 'childs_pose_backonheels', 'childs_pose_throughshoulders', 'childs_pose_end']
#state_descriptions = [
#     'Waiting to Start',
#     'On all fours',
#     'Back on heels',
#     'Full extension',
#     'Finished' ]

state_names = ['crawlouts_begin', 'crawlouts_standing', 'crawlouts_crawling', 'crawlouts_plank', 'crawlouts_finish']
state_descriptions = [
	'Waiting to Start',
	'Standing',
	'Crawling',
	'Plank',
	'Finished'
]

states = {'state_names': state_names, 'state_descriptions': state_descriptions}

# open output file for writing
with open('states.json', 'w') as filehandle:
    json.dump(states, filehandle)
