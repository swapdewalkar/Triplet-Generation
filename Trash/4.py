#!/usr/bin/env python
# coding: utf-8

# In[12]:
#Wordnet Relation

import functions as F

from nltk.corpus import wordnet as wn1
from nltk.corpus import stopwords


# In[8]:

dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("Start",time_t)

vocab_file='vocab'
vocab=F.load_to_file(vocab_file)
wordnet_realtion_file='wordnet_relation'


# In[22]:


stop=stopwords.words('english')


# In[3]:


# If strong 3
# If weak 1 : w1 present in w2 definition
# If weak 2 : w2 present in w1 definition

def IsStrong(syn1,syn2,w1,w2):
    w1_def=""
    for t in syn1:
        w1_def += t.definition()
    w2_def=""
    for t in syn2:
        w2_def += t.definition()
    flag1=w1_def.find(w2)
    flag2=w2_def.find(w1)
    if flag1>-1 and flag2>-1:
        return 3
    elif flag1>-1:
        return 2
    elif flag2>-1:
        return 1
    else:
        return 0

def get_relation(word1,word2):
    rel=[]
    syn1=wn1.synsets(word1) 
    syn2=wn1.synsets(word2) 
#     print(syn1)
#     print(syn2)
    inter=[1 for s1 in syn1 for s2 in syn2 if s1==s2]  
    if len(inter)>0:
        rel.append('synset')
    is_strong=IsStrong(syn1,syn2,word1,word2)
    if is_strong==3:
        rel.append('strong')
    elif is_strong>0:
        rel.append('weak')

    hyponyms1  = [] 
    for t in syn1:
        hyponyms1 += t.hyponyms()
    hypernyms1 = []
    for t in syn1:
        hypernyms1 += t.hypernyms()
    holonyms1 = []
    for t in syn1:
        holonyms1  += t.part_holonyms()
#     print(hyponyms1)
    inter_hypo=[1 for s1 in hyponyms1 for s2 in syn2 if s1==s2]  
    inter_hyper=[1 for s1 in hypernyms1 for s2 in syn2 if s1==s2] 
    inter_holo=[1 for s1 in holonyms1 for s2 in syn2 if s1==s2] 
    if len(inter_hypo)>0:
        rel.append('hyponym') 
    if len(inter_hyper)>0:
        rel.append('hypernym')
    if len(inter_holo)>0:
        rel.append('holonym')    
    return rel


# In[24]:


d={}
count=1
for w1 in vocab:
    print(count)
    countj=0
    d[w1]={}
    if w1 not in stop and len(w1)>2:
        for w2 in vocab:
            countj += 1
            if w1!=w2 and w2 not in stop and len(w2)>2:
                rel=get_relation(w1,w2)
                if len(rel)>0:   
                    d[w1][w2]=rel
        print(count,countj)
        count += 1


# In[9]:


F.save_to_file(wordnet_realtion_file,d)


# In[10]:


a=F.load_to_file(wordnet_realtion_file)


# In[11]:


print(a)


# In[ ]:



dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("END",time_t)
