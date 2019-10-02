import random
import requests
from spellcheck import spellcheck
import re
import json
from pprint import pprint
import Levenshtein
import nltk

from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


with open('OntologyToLemmas.json', 'r') as fp:
	    key2lemmas=json.load(fp)

# print(key2lemmas)
with open('LemmasToOntology.json', 'r') as fp:
	    lemma2key=json.load(fp)

def check_lemmas(user_input):
	wordnet_lemmatizer=WordNetLemmatizer()
	input_lemma=wordnet_lemmatizer.lemmatize(user_input)
	if input_lemma in lemma2key:
		return lemma2key[input_lemma]
	else:
		lemma_syns=wn.synsets(input_lemma)[0]
		max_similarity=0
		closest_lemma=None
		for word in key2lemmas.keys():
			for lemma in key2lemmas[word]:
				comparison_syns=wn.synsets(lemma)[0]
				similarity=comparison_syns.path_similarity(lemma_syns)
				if not similarity:
					similarity=0
				if similarity>max_similarity:
					max_similarity=similarity
					closest_lemma=word
		if max_similarity>=0.3:
			return closest_lemma
		else:
			return None

