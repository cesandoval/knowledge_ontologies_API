import random
import requests
from spellcheck import spellcheck
import re
import json
from pprint import pprint
from Scraper import *
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import sys
import spacy
import itertools
import numpy as np
import check_lemmas
from sklearn.metrics.pairwise import cosine_similarity
from config import L1,L2,L3

full_ontology=L1+L2+L3
full_list=set(full_ontology)
nlp = spacy.load('en_core_web_md')


with open ('vectors_map.json','r') as f:
	vector_dict=json.load(f)

keyword=None
while keyword!= 'N':
	if keyword== 'N':
		break


	keyword=input('Word to classify:')

	checked=spellcheck(keyword)

	##check if it is an existing key in the ontology
	if keyword in full_list:
		print(keyword)
		continue
	else:
		secondary_check=check_lemmas.check_lemmas(checked)
		if secondary_check:
			print(secondary_check)
			continue

	keyword_doc = list(nlp.pipe(checked,
	  batch_size=10000,
	  n_threads=1))


	if keyword_doc[0].has_vector:
		keyword_vector=np.array([keyword_doc[0].vector])
	else:
		spacy.vocab[0].vector

	intermidiate_results=[]
	best=None
	best_similarity=0
	## First level

	l1_keys=treeL1.keys()
	arrays=[]
	order=[]
	for k in l1_keys:
		order.append(k)
		arrays.append(np.array(vector_dict[k]))

	simple_sim = cosine_similarity(keyword_vector, arrays)
	topic_idx = simple_sim.argmax(axis=1)[0]
	best_similarity=np.amax(simple_sim)
	result=order[topic_idx]
	best=result
	# print('r1: ', result )

	##second LEvel
	l2_keys=[word for word in treeL1[result] if word in treeL2]
	# print('choices: ', l2_keys)
	arrays=[]
	order=[]
	for k in l2_keys:
		order.append(k)
		arrays.append(np.array(vector_dict[k]))

	simple_sim = cosine_similarity(keyword_vector, arrays)
	topic_idx = simple_sim.argmax(axis=1)[0]
	result=order[topic_idx]
	maxv=np.amax(simple_sim)
	if (maxv>=best_similarity):
		best_similarity=maxv
		best=result
	# print('r2: ',result)


	options=[ word for word in treeL2[result] if word in full_ontology]
	# print('choices: ', options)
	#print('options',options)
	arrays=[]
	for k in options:
		#print(k,len(vector_dict))
		arrays.append(np.array(vector_dict[k]))

	simple_sim = cosine_similarity(keyword_vector, arrays)
	topic_idx = simple_sim.argmax(axis=1)[0]
	maxv=np.amax(simple_sim)
	result=options[topic_idx]
	if (maxv>=best_similarity):
		best_similarity=maxv
		best=result
	print( 'classified as: '+best)


