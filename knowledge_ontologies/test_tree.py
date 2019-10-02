import spacy
import itertools
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('en_core_web_md')


# L1=['technology', 'society', 'entertainment', 'culture', 'humanity', 'science']
# L2=['gastronomy', 'popular culture', 'leisure activities',  'religion', 'humanities', 'philosophy', 'mathematics', 'natural and physical sciences', 'health', 'engineering', 'industry and logistics', 'information science', 'social science', 'economy', 'politics', 'education', 'literature', 'performing arts', 'visual arts', 'design', 'fashion', 'gastronomy', 'popular culture',  'leisure activities']
# L3=['communication and transport',  'industries', 'construction', 'energy', 'electrical engineering',  'mechanical engineering',  'aerospace', 'chemical engineering', 'nanotechnology', 'forensic science',  'civil engineering', 'earth science', 'agriculture', 'biotechnology', 'health sciences', 'medicine', 'biology', 'chemistry', 'physics', 'mathematics', 'logic', 'philosophy', 'languages', 'comparitive literature', 'history', 'classical studies', 'archaeology', 'irrelegion', 'sprituality', 'religion', 'geography and places', 'travel', 'recreation', 'games', 'sports', 'celebrities', 'corporate branding', 'pop media', 'pop music', 'meals', 'food preparation', 'food and drink', 'cuisines', 'costumes', 'fashion design', 'landscape design', 'city planning', 'architecture', 'product design', 'graphic design', 'painting', 'sculpture', 'installation', 'new media art', 'film', 'music', 'dance', 'theater', 'poetry', 'fiction', 'non-fiction', 'education', 'communication', 'politics', 'political science', 'rights', 'law', 'military', 'business', 'economics', 'physcology', 'sociology', 'archaeology', 'anthropology', 'future studies', 'semiotics', 'linguistics', 'neuroscience', 'communication', 'knowledge classification', 'information science', 'computing']
L1=['sciences', 'earth', 'life', 'humanity', 'community', 'leisure', 'entertainment', 'arts', 'society', 'industry']

L2=['sports', 'hobbies', 'places', 'events', 'sharing', 'religion', 'history', 'philosophy', 'brain science', 'health', 'medical specialties', 'biomedical sciences', 'biology', 'ecology', 'earth sciences', 'engineering', 'physical sciences', 'computation', 'social science', 'internet and logistics', 'industries', 'economy', 'military', 'law', 'politics', 'education', 'literature', 'performing arts',  'film', 'visual arts', 'design', 'fashion', 'food and drink', 'pop music', 'celebrities', 'adult']
L3=['gaming', 'sports', 'recreation and fitness', 
'pets', 'crafts', 'home improvement',  'cars and vehicles',  
'maps and places', 'travel', 
'holidays', 'weather', 'local events', 'crime and prosecution',  'people', 'social network', 'search engine', 
'email', 'chats and forums',  'file sharing',  'dating',  'philanthropy', 'genealogy', 
'death', 'religion', 'spirituality', 
'classical studies', 'history', 
'comparative literature', 'philosophy', 'languages', 
'neuroscience', 'physchiatry', 'physcology', 
'mental health',  'alternative and natural medicine', 
'personal health data', 
'pharmacy and health products', 'dental health', 'senior health', 'childs health',  'reproductive health',  'diseases', 
'medical imaging',  'medical specialties', 
'biotechnology and pharmaceuticals', 
'immunology', 'molecular biology', 'cell biology'
'animal and plant biology',  'veterinary', 
 'food and science technology', 'agriculture',  'climate', 'environment', 
 'geology', 'metals and mining', 'materials', 
 'nanotechnology', 'aerospace', 'civil engineering', 'mechanical engineering',  'electrical engineering', 
 'chemistry', 'physics', 'mathematics', 
 'computer and electronics', 'computer science',  'linguistics', 'anthropology', 'sociology', 'communication science', 
 'telecommunications', 'web administration', 
 'shipping', 'transport', 'construction and maintenance', 'real estate',  'industrial goods and services', 'energy', 'associations', 
 'marketing and advertising', 'economics', 'finance', 'business', 
 'military', 'war', 
 'law', 'criminal justice', 
 'government', 'political science', 'immigration and visas', 'daily politics', 
 'educational policy', 'careers', 'jobs and employment', 'business training', 'schools', 
 'publishing', 'non-fiction', 'fiction', 'poetry', 
 'theater', 'dance', 'music', 
 'film', 'animation and comics', 
 'photography', 'new media and installation art', 'sculpture', 'painting', 
 'web and graphic design', 'product design', 'architecture', 'city planning', 'landscape and garden', 
 'fashion', 'apparel', 'beauty', 
 'food preparation', 'places to eat and drink', 
 'pop music', 
 'celebrities', 
 'adult', 'gambling'
]
all_levels=L1+L2+L3

treeL1={'sciences':['engineering', 'social science', 'computation', 'physical sciences', 'investigation', 'research', 'empirical', 'data', 'experiment', 'matter'], 
	'earth':['earth sciences', 'engineering', 'earth sciences', 'ecology', 'biology', 'planet'], 
	'life':['biology', 'biomedical', 'medicine', 'health', 'evolution', 'growth', 'being', 'organism', 'viable'], 
	'humanity':['health', 'brain science', 'philosophy', 'history', 'religion', 'culture', 'rational', 'mind', 'consciousness', 'compassion'], 
	'community':['religion', 'sharing', 'events', 'neighbors', 'local'], 
	'leisure':['sports', 'hobbies', 'places', 'events', 'relaxation', 'relief', 'respite'], 
	'entertainment':['sports', 'adult', 'celebrities', 'pop music',  'food and drink', 'fashion',  'show',  'performance',  'entertainment',  'spectacle'], 
	'arts':['fashion',  'design',  'visual arts', 'film', 'performing arts', 'literature', 'creative', 'expression'], 
	'society':['military', 'law', 'politics', 'education', 'literature',  'education', 'government', 'bureaucracy'], 
	'industry':['military', 'economy', 'industries', 'internet and logistics', 'production', 'fabrication']

}
# treeL2={'gastronomy':['meals', 'food preparation', 'food and drink', 'cuisines', 'restuarant', 'food', 'taste'], 
# 'popular culture':['sports', 'celebrities', 'corporate branding', 'pop media', 'pop music', 'fame', 'media', 'icons'], 
# 'leisure activities':['geography and places', 'travel', 'recreation', 'games', 'sports', 'competition', 'tourism', 'pleasure'],  
# 'religion':['irrelegion', 'sprituality', 'religion', 'atheism', 'belief', 'creed'], 
# 'humanities':['languages', 'comparative literature', 'history', 'classical studies', 'archaeology', 'classics'], 
# 'philosophy':['philosophy'], 
# 'mathematics':['mathematics', 'logic', 'proof', 'numbers'], 
# 'natural and physical sciences':['biology', 'chemistry', 'physics', 'mathematics'], 
# 'health':['agriculture', 'biotechnology', 'health sciences', 'medicine', 'diet', 'care'], 
# 'engineering':['energy', 'electrical engineering',  'mechanical engineering',  'aerospace', 'chemical engineering', 'nanotechnology', 'forensic science',  'civil engineering', 'earth science', 'agriculture', 'innovation', 'design', 'construction'], 
# 'industry and logistics':['communication and transport',  'industries', 'construction', 'energy', 'manufacturing'], 
# 'information science':['communication', 'knowledge classification', 'information science', 'computing', 'computation', 'data', 'classification'], 
# 'social science':['physcology', 'sociology', 'archaeology', 'anthropology', 'future studies', 'semiotics', 'linguistics', 'neuroscience'], 
# 'economy':['business', 'economics', 'finance', 'currency', 'inflation'], 
# 'politics':['politics', 'political science', 'rights', 'law', 'military', 'business', 'president', 'democracy', 'leader', 'constitution'], 
# 'education':['education', 'learning', 'study', 'learn'], 
# 'literature':['poetry', 'fiction', 'non-fiction', 'education', 'books', 'read', 'text'], 
# 'performing arts':['film', 'music', 'dance', 'theater', 'creative', 'arts'], 
# 'visual arts':['painting', 'sculpture', 'installation', 'new media art', 'film', 'visual', 'arts'], 
# 'design':['landscape design', 'city planning', 'architecture', 'product design', 'graphic design', 'drawing', 'plan', 'sketch'], 
# 'fashion':['costumes', 'fashion design', 'modern', 'style']
treeL2={'sports':['gaming', 'sports', 'recreation and fitness', 'ball', 'score', 'compete', 'team', 'athletics', 'exercise'], 
'hobbies':['pets', 'crafts', 'home improvement',  'cars and vehicles', 'free time', 'fun'], 
'places':['maps and places', 'travel', 'destinations', 'tourism', 'locations', 'marvels', 'attractions'], 
'events':['holidays', 'weather', 'local events', 'crime and prosecution',  'people', 'social network', 'search engine', 'event', 'occasion', 'news'], 
'sharing':['email', 'chats and forums',  'file sharing',  'dating',  'philanthropy', 'genealogy', 'communication', 'social', 'exchange'], 
'religion':['death', 'religion', 'spirituality', 'soul', 'beliefs', 'existence'], 
'history':['classical studies', 'history', 'timeline', 'archive'], 
'philosophy':['comparative literature', 'philosophy', 'languages', 'thought', 'reasoning'], 
'brain science':['neuroscience', 'physchiatry', 'physcology', 'brain', 'neuron'], 
'health':['mental health',  'alternative and natural medicine', 
'personal health data', 
'pharmacy and health products', 'dental health', 'senior health', 'childs health',  'reproductive health',  'diseases', 'pediatrics'], 
'medical specialties':['medical imaging',  'medical specialties', 'analysis', 'diagnostics'], 
'biomedical sciences':['biotechnology and pharmaceuticals', 
'immunology', 'molecular biology', 'cell biology' ], 
'biology':['cell biology', 'animal and plant biology',  'veterinary', 'life', 'nature'], 
'ecology':['food and science technology', 'agriculture',  'climate', 'environment', 'ecology',  'ecosystem'], 
'earth sciences':['geology', 'metals and mining', 'materials'], 
'engineering':['nanotechnology', 'aerospace', 'civil engineering', 'mechanical engineering',  'electrical engineering', 'innovation', 'construction', 'solution', 'materials'], 
'physical sciences':['chemistry', 'physics', 'mathematics'], 
'computation':['computer and electronics', 'computer science', 'algorithms', 'computer', 'processing'], 
'social science':['linguistics', 'anthropology', 'sociology', 'communication science'], 
'internet and logistics':['telecommunications', 'web administration', 'internet', 'web'], 
'industries':['telecommunications', 'web administration', 
 'shipping', 'transport', 'construction and maintenance', 'real estate',  'industrial goods and services', 'energy', 'associations'], 
'economy':['marketing',  'advertising', 'economics', 'finance', 'business', 'money', 'inflation', 'deflation'], 
'military':['military', 'war', 'weapons', 'navy', 'army', 'soldiers', 'troops'], 
'law':['law', 'criminal justice', 'justice', 'crime', 'prosecution', 'defense'], 
'politics':['government', 'political science', 'immigration and visas', 'daily politics', 'politics', 'news'], 
'education':['educational policy', 'careers', 'jobs and employment', 'business training', 'schools', 'learning', 'teaching', 'knowledge'], 
'literature':['publishing', 'non-fiction', 'fiction', 'poetry', 'writing', 'reading'], 
'performing arts':['theater', 'dance', 'music', 'performance', 'arts', 'drama'], 
 'film':['film', 'animation and comics', 'movies', 'drama'], 
 'visual arts':['photography', 'new media and installation art', 'sculpture', 'painting', 'media', 'art'], 
 'design':['web and graphic design', 'product design', 'architecture', 'city planning', 'landscape and garden', 'design'], 
 'fashion':['fashion', 'apparel', 'beauty', 'trends', 'modern'], 
 'food and drink':['food preparation', 'places to eat and drink', 'restaurants', 'bars', 'cuisine'], 
 'pop music':['pop music', 'mainstream'], 
 'celebrities':['celebrities', 'fame'], 
 'adult':['adult', 'gambling', 'mature']}



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

