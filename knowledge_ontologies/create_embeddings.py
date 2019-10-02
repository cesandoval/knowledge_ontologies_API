import spacy
import itertools
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from config import *
nlp = spacy.load('en_core_web_md')

all_levels=L1+L2+L3

## This script creates word embeddings for all of the categories using Spacys pretrained word vectors

vectors={}




def key_vectors():
	topic_keywords=[]
	topic_labels=[]

	for label in L1:
		topic_labels.append(label)
		add_str=''
		for el in treeL1[label]:
			add_str+=el+' '
			try:
				for el2 in treeL2[el]:
					add_str+=el2 + ' '
			except:
				continue
		topic_keywords.append(add_str)

	for label in L2:
		topic_labels.append(label)
		add_str=''
		for el in treeL2[label]:
			add_str+=el+' '
		topic_keywords.append(add_str)

	for label in L3:
		topic_labels.append(label)
		topic_keywords.append(label)

	topic_docs = list(nlp.pipe(topic_keywords,  batch_size=10000,  n_threads=3))
	topic_vectors = np.array([doc.vector if doc.has_vector else spacy.vocab[0].vector for doc in topic_docs])

	return topic_vectors




keys=key_vectors()

for k in range(len(keys)):
	vectors[all_levels[k]]=keys[k].tolist()

with open('vectors_map.json',  'w') as fp:
    json.dump(vectors,  fp)

