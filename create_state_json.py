import json

# define list with values
states = ['childs_pose_start', 'childs_pose_allfours', 'childs_pose_backonheels', 'childs_pose_throughshoulders', 'childs_pose_end']

# open output file for writing
with open('states.json', 'w') as filehandle:
    json.dump(states, filehandle)
