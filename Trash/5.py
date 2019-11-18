#!/usr/bin/env python
# coding: utf-8

# In[5]:
#Positive and NUM

import functions as F

dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("Start",time_t)

vocab_file='vocab'
dp_relation_file='dp_relation'
dp_triplet_file='dp_triplets'
wordnet_triplet_file='wordnet_relation'
occ_triplet_file='occurrence'
word_to_index_file='word_to_index'
index_to_word_file='index_to_word'

wn_num_file='wn_num'
occ_num_file='occ_num'
dp_num_file='dp_num'
occ_num_dups_file='occ_num_dups'
relation_to_index_file='relation_to_index'
index_to_relation_file='index_to_relation'

all_relations='all_relations'


# In[6]:


vocab = F.load_to_file(vocab_file)

word_to_index = F.load_to_file(word_to_index_file)
index_to_word = F.load_to_file(index_to_word_file)

dp_relation = F.load_to_file(dp_relation_file)
dp_triplet = F.load_to_file(dp_triplet_file)

wordnet_triplet = F.load_to_file(wordnet_triplet_file)





# import os
# os.listdir(F.folder+"occurences")
# files=os.listdir(F.folder+"occurences")
occ={}
# flag=1;
# for f in files:
#     print(f)
#     if flag:
#         occ = F.load_to_file("occurences/"+f)
#         flag=0
#     else:
#         temp_occ=F.load_to_file("occurences/"+f)
#         for k in occ:
#             occ[k] += temp_occ[k]

wordnet_relation=['synset','hyponym','hypernym','holonym','strong','weak']
dp_relation=['advmod','amod','appos','compound','conj','fixed','flat','doeswith','list','nmod','nummod','orphan','reparandum']

# In[7]:


relations=dp_relation + wordnet_relation + list(occ.keys())
relation_to_index={}
index_to_relation={}
for k,v in enumerate(relations):
    relation_to_index[v]=k
    index_to_relation[k]=v
F.save_to_file(relation_to_index_file,relation_to_index)
F.save_to_file(index_to_relation_file,index_to_relation)


# In[8]:


relation_to_index=F.load_to_file(relation_to_index_file)
index_to_relation=F.load_to_file(index_to_relation_file)


# In[5]:


print(relation_to_index)
print(index_to_relation)


# In[9]:


dp_number_triple=[]
dp_relation_num=[relation_to_index[r] for r in dp_relation]
count=0
for dp_triple in dp_triplet:
    try:
        a,b,c=dp_triple
        a=word_to_index[a]
        b=relation_to_index[b]
        c=word_to_index[c]
        dp_number_triple.append((a,b,c))
    except:
        print(c)
        count +=1
len(dp_number_triple)


# In[10]:


wn_number_triple=[]
wn_relation_num=[relation_to_index[r] for r in wordnet_relation]
for w1 in wordnet_triplet:
    for w2 in wordnet_triplet[w1]:
        a=word_to_index[w1]
        b=word_to_index[w2]
        for c in wordnet_triplet[w1][w2]:
            c=relation_to_index[c]
            wn_number_triple.append((a,c,b))
len(wn_number_triple)


# In[11]:


#All
occ_number_triple=[]
occ_relation_num=[relation_to_index[r] for r in list(occ.keys())]
for r in occ:
    c=relation_to_index[r]
    for a,b in occ[r]:
        occ_number_triple.append((a,c,b))
len(occ_number_triple)


# In[12]:


#without duplicates
occ_number_triple_without_duplicate={}
occ_relation_num_without_duplicate=[relation_to_index[r] for r in list(occ.keys())]
for r in occ:
    if r<10 and r>-10:
        c=relation_to_index[r]
        print(r,c)
        l=0;
        for a,b in occ[r]:
    #         if (a,c,b) not in occ_number_triple_without_duplicate:
                occ_number_triple_without_duplicate[(a,c,b)]=1
        print(len(occ_number_triple_without_duplicate)-l)
print(list(occ_number_triple_without_duplicate.keys())[:10])
print(len(list(occ_number_triple_without_duplicate.keys())))
occ_number_triple_without_dup=list(occ_number_triple_without_duplicate.keys())


# In[13]:


F.save_to_file(all_relations,relations)
print(len(relations))
print(len(wn_number_triple))
print(len(dp_number_triple))
print(len(occ_number_triple))
print(len(occ_number_triple_without_duplicate))


# In[14]:


print(index_to_relation)


# In[15]:


F.save_to_file(wn_num_file,wn_number_triple)
F.save_to_file(occ_num_file,occ_number_triple)
F.save_to_file(dp_num_file,dp_number_triple)
F.save_to_file(occ_num_dups_file,occ_number_triple_without_dup)


# In[16]:


print(len(wn_number_triple),len(occ_number_triple),len(dp_number_triple))


# In[17]:


positive_table={}
total_triple=wn_number_triple + dp_number_triple + occ_number_triple_without_dup
for triple in total_triple:
    a,b,c=triple
    if a not in positive_table:
        positive_table[a]={}
    if b not in positive_table[a]:
        positive_table[a][b]=[c]
    else:
        positive_table[a][b].append(c)
        
F.save_to_file('Positive_Table',positive_table)

dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("END",time_t)