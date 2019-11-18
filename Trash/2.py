#!/usr/bin/env python
# coding: utf-8

# In[1]:
#Occurence

import functions as F


dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("Start",time_t)

data_file='updated_words'
vocab_file='vocab'
w2i_file='word_to_index'
i2w_file='index_to_word'
occurrence_data_file='occurrence'


data=F.load_to_file(data_file)
vocab=F.load_to_file(vocab_file)
word_to_index=F.load_to_file(w2i_file)
index_to_word=F.load_to_file(i2w_file)
print(word_to_index)
print(index_to_word)
print(len(vocab),len(data))

data_index=[word_to_index[w] for w in data ]

unknown_id=word_to_index['unknown']


occurrence={}
window=2
print("Words:",len(data_index))
for i in range(-window,window+1):
	occurrence[i]=[]


for c in range(len(data_index)):
	print(c)
	start=max(0,c-window)
	end=min(len(data_index)-1,c+window)
#     print(start,end)
	if data_index[c]!=unknown_id:
		for j in range(start,end+1):
			if c!=j and data_index[j]!=0:
#                 print(j,c)
				occurrence[j-c].append((data_index[c],data_index[j]))
	# if(c%10000000==9999999):
	if(c%10000000==9999999):
		F.save_to_file("occurences/"+occurrence_data_file+str((c/10000000)+1),occurrence)
		for i in range(-window,window+1):
			occurrence[i]=[]

for k in occurrence:
	print(k,len(occurrence[k]))


# data=F.load_to_file(occurrence_data_file)


# In[ ]:




dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("END",time_t)