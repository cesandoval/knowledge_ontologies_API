import random
import requests
from spellcheck import spellcheck
import re
import json
from pprint import pprint
import nltk
from config import *
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


data={}

ontologyl1=L1

ontologyl2=L2
ontologyl3=L3
full_ontology=ontologyl1+ontologyl2+ontologyl3



def main():
	nltk.download('wordnet')
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	wordnet_lemmatizer=WordNetLemmatizer()
	key2lemmas={}
	lemma2key={}
	synset_info={}

	for phrase in full_ontology:
		tokens=nltk.word_tokenize(phrase)
		tagged_words=nltk.pos_tag(tokens)
		filtered_fillers=[]
		for text,tag in tagged_words:
			if tag!='CC' and tag!='TO' and tag!='IN':
				filtered_fillers.append(text)
		for word in filtered_fillers:
			try:
				#phrase could already be in dict
				lemma=wordnet_lemmatizer.lemmatize(word)
				parent=wn.synsets(lemma)[0]
				related_lemmas=parent.lemmas()
				related_lemma_names=parent.lemma_names()
				if phrase not in key2lemmas:
					key2lemmas[phrase]=related_lemma_names
				else:
					key2lemmas[phrase].extend(related_lemma_names)
				for lemma in related_lemma_names:
					lemma2key[lemma.lower()]=phrase

			except:
				print ('error',word)
				lemma=wordnet_lemmatizer.lemmatize(word)
				lemma2key[lemma.lower()]=phrase

	with open('OntologyToLemmas.json', 'w') as fp:
	    json.dump(key2lemmas, fp)


	with open('LemmasToOntology.json', 'w') as fp:
	    json.dump(lemma2key, fp)

if __name__=="__main__":
	main()