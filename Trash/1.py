#!/usr/bin/env python
# coding: utf-8

# In[1]:
#Preprossing and DP

import functions as F
sents_file_name='sents'
words_file_name='words'
updated_words_file_name='updated_words'
vocab_file='vocab'
w2i_file='word_to_index'
i2w_file='index_to_word'


# In[2]:
dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("Start",time_t)



sentences=F.load_to_file(sents_file_name)
# # sent_data=F.remove_special_from_sent_data(sent_data)
# # F.save_to_file('filtered_sent_data',sent_data)
# sent_data_filter=F.load_to_file('filtered_sent_data')
# sent_data=sent_data_filter

print("Sentence:",len(sentences))
# print("Sentence:",sentences)
# print(sent_data)
# # sent_data=sent_data[:10000]
import threading 
NO_OF_THREADS=25
triplets_dict={}
F.count=0
from nltk.parse.stanford import StanfordDependencyParser
path_to_jar = '/home/cs17mtech11004/stanford-parser-full-2018-02-27/stanford-parser.jar'
path_to_models_jar = '/home/cs17mtech11004/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

def func(i):
    start = int(i *( len(sentences)/NO_OF_THREADS ))
    end= int(min(((i+1) * (len(sentences)/NO_OF_THREADS )),len(sentences)) - 1)
    # print(start,end)
	# triplets_dict[i]=F.genrate_triplet(sent_data[start:end])
    F.genrate_triplet(i,sentences[start:end],dependency_parser)

t = [ threading.Thread(target=func, args=(i,)) for i in range(NO_OF_THREADS) ] 
for temp in t:
    temp.start() 
for temp in t:
    temp.join() 


# F.save_to_file('dp_triplets_dict',triplets_dict)

dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("End",time_t)




