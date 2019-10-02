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
'immunology', 'molecular biology', 'cell biology', ], 
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

threshold=0.3

##list of categories by level 
L1=['sciences', 'earth', 'life', 'humanity', 'community', 'leisure', 'entertainment', 'arts', 'society', 'industry']

L2=['sports', 'hobbies', 'places', 'events', 'sharing', 'religion', 'history', 'philosophy', 'brain science', 'health', 'medical specialties', 'biomedical sciences', 'biology', 'ecology', 'earth sciences', 'engineering', 'physical sciences', 'computation', 'social science', 'internet and logistics', 'industries', 'economy', 'military', 'law', 'politics', 'education', 'literature', 'performing arts',  'film', 'visual arts', 'design', 'fashion', 'food and drink', 'pop music', 'celebrities', 'adult']

##List of cateogries in third level
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
'immunology', 'molecular biology', 'cell biology','animal and plant biology',  'veterinary', 
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