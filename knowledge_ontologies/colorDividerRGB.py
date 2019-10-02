from config import *

##This code creates a 1xn flattened list of all the labels in order that are then assigned rgb codes
def flatten(word):
	answer=[]
	childrenl1=list(filter(lambda x: x in L2,treeL1[word]))
	for i in range(0,len(childrenl1)//2):
		childrenl2=list(filter(lambda x: x in L3,treeL2[childrenl1[i]]))
		for label in childrenl2[0:len(childrenl2)//2]:
			answer.append(label)
		answer.append(childrenl1[i])
		for label in childrenl2[len(childrenl2)//2:]:
			answer.append(label)
	answer.append(word)
	for child1 in childrenl1[len(childrenl1)//2:]:
		childrenl2=list(filter(lambda x: x in L3,treeL2[child1]))
		for label in childrenl2[0:len(childrenl2)//2]:
			answer.append(label)
		answer.append(childrenl1[i])
		for label in childrenl2[len(childrenl2)//2:]:
			answer.append(label)
	return answer

all_flattened=[]
for word in L1:
	all_flattened+=flatten(word)

import colorsys
N = len(all_flattened)
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
RGB_vals=list(RGB_tuples)
color_dict={}
for i in range(len(RGB_vals)):
	color_dict[all_flattened[i]]=RGB_vals[i]


import json
with open('color_map.json',  'w') as fp:
    json.dump(color_dict,  fp)



	