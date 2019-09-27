# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import json

from bs4 import BeautifulSoup
import urllib
import nltk


#%%
nltk.download()


#%%
# The number in the brackets ([]), indicates the number of records to process. 
# Change the number for a larger sample size. 
title_dict = {}
with open('history.json') as json_data:
    d = json.load(json_data)
    for entry in d:
        if entry['title'] != '':
            if entry['url'] not in title_dict:
                title_dict[entry['url']] = entry['title']
        else:
            try:
                r = urllib.urlopen(entry['url']).read()
                soup = BeautifulSoup(r)
                title_dict[entry['url']] = soup.title.string
            except: 
                pass


#%%
title_dict.values()


#%%
from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords

from gensim import corpora, models, similarities
import numpy as np

import json
import csv


#%%
titles = title_dict.values()

clean_titles = []
for ctitle in titles:
    if ctitle != None:
        clean_titles.append(ctitle)
dictionary = corpora.Dictionary([word.lower().encode('utf-8').split(' ') for word in clean_titles])
browsing_titles = [word.lower().encode('utf-8').split(' ') for word in clean_titles]


#%%
with open("wordsEn.txt") as word_file:
    english_words = set(word.strip().lower() for word in word_file)

def is_english_word(word):
    return word.lower() in english_words

only_english_ids = [word[1] for word in dictionary.token2id.iteritems() if not is_english_word(word[0])]      


#%%
# get a list of stop words from the nltk library
stoplist = stopwords.words('english')

# DICTERATOR: remove stop words and words that appear only once 
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]

# filter the tokens from the corpora dict
dictionary.filter_tokens(stop_ids + once_ids + only_english_ids)


#%%
# remove gaps in id sequence after words that were removed
dictionary.compactify() 
dictionary


#%%
# given a dictionary and a list of ids, get the words that correspond to the ids
def get_singles(dictionary, ids):
    for word_id in ids:
        yield dictionary.get(word_id)

# eliminate the words that appear only once        
def filter_singles(singles, texts):
    for text in texts:
        new_list = []
        for word in text:
            if word not in singles:
                new_list.append(word)
        yield new_list


#%%
singles = get_singles(dictionary, once_ids)
filtered_texts = filter_singles(singles, [word.lower().encode('utf-8').split(' ') for word in clean_titles])


#%%
# # Create Bag of words
mm = [dictionary.doc2bow(text) for text in filtered_texts]


#%%
# define the number of topics for the classification
num_topics = 10

# Trains the LDA models with the corpus and dictionary previously created
lda = models.ldamodel.LdaModel(corpus=list(mm), id2word=dictionary, num_topics=num_topics, 
                               update_every=1, chunksize=10000, passes=10, iterations=50)


#%%
# prints all groups and their main words
lda.print_topics(num_topics=num_topics, num_words=25)


#%%
# get a list of processed topics obtained by training an LDA model, and return them as individual lists of topics and frequencies
def parse_topics(filepath):
    with open(filepath, 'rU') as f:
        reader = list(csv.reader(f))
        header = reader[0]
        reader.pop(0)
        topics = []
        freqs = []
        for row in reader:
            freq = []
            topic = []
            row.pop(0)
            for ind, element in enumerate(row):
                if ind%2 == 0:
                    try: 
                        fr = row[ind+1]
                    except: 
                        fr = ''
                    if fr != '':
                        topic.append(element)
                        freq.append(row[ind+1])
            topics.append(topic)
            freqs.append(freq)
        return topics, freqs


#%%
topics, frequencies = parse_topics('knowledge_topic_classification.csv')
topics, frequencies


#%%
# Assigns the topics to the documents in corpus
lda_corpus = lda[mm]
threshold = 1/float(num_topics)


#%%
# given a corpus trained with the LDA classifier, and a threshold, classify the browsing history into the groups 
def classify(lda_corpus, texts, cluster_num, threshold, words=None, frequencies=None):
    for i,j in zip(lda_corpus, texts):
        try: 
            if i[cluster_num][1] > threshold :
                classified_list = [j, words[cluster_num], frequencies[cluster_num]]
                yield classified_list
        except: pass


#%%
# function that takes the topic classification of a given tweet, and other data of the tweets and writes a new json to be spatially joined
def topic_to_json(topic_num, topics, frequencies):
    for i, record in enumerate(classify(lda_corpus, clean_titles, topic_num, threshold, topics, frequencies)):  
        title, topic, frequency = record
        with open('topics/%stopic_history.json' %(str(topic_num)+'_'+str(i)), 'w') as f:
            f.write( json.dumps({'id': str(topic_num)+'_'+str(i), 'title':title, 'topic':topic, 'frequency':frequency}))
            #print 'wrote tweet %s' %(tid)


#%%
# for every topic group, write json files for every tweet
for topic_num in np.arange(num_topics):#lda_corpus, jsons_to_mm_tuple(twi_path), topic_num, threshold, num_topics): 
    topic_to_json(topic_num, topics, frequencies)


#%%
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir('topics') if isfile(join('topics', f))]
count_topics = {}
for file in onlyfiles:
    if file.endswith('.json'):
        with open('topics/'+file, 'r') as f:
            curr_record = json.load(f)
            curr_topics = curr_record['topic']
            for curr_topic in curr_topics:
                if curr_topic not in count_topics:
                    count_topics[curr_topic] = 0
                else: 
                    count_topics[curr_topic] += 1


count_topics


#%%
import seaborn as sns
import matplotlib.pyplot as plt

x_vals = count_topics.keys()
y_vals = count_topics.values()


y_pos = np.arange(len(x_vals))
 
plt.bar(y_pos, y_vals, align='center', alpha=0.5)
plt.xticks(y_pos, x_vals)
plt.ylabel('Number of Ocurrences')
plt.title('Domains of Knowledge')
 
plt.show()


