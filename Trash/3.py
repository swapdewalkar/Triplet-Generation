#!/usr/bin/env python
# coding: utf-8

#     # Filter DP triple based on vocab

# In[1]:

#DP Dict to Triplet
import functions as F

dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("Start",time_t)

vocab_file='vocab'
vocab=F.load_to_file(vocab_file)

# triplets_dict_file='dp_triplets_dict'
# dp_triplets=F.load_to_file(triplets_dict_file)

final_triplet_file='dp_triplets'
dp_relation_file='dp_relation'



# #Concatenate all triplets from threads
# final_triplet_with_pos=[]
# print(len(dp_triplets))
# for m in dp_triplets:
# #     print(len(triplets[m]))
#     for n in dp_triplets[m]:
#         for o in n:
# #             print(len(o))
#             final_triplet_with_pos.append(o)
# print(len(final_triplet_with_pos))
import os
os.listdir(F.folder+"dp_data_pos")
files=os.listdir(F.folder+"dp_data_pos")

relation=[]
final_triplet=[]
for f in files:
    triplet_data=F.load_to_file("dp_data_pos/"+f)
    # print(triplet_data)
    #Find H R T
    for sent in triplet_data:
        for t in sent:
            (H,HPOS),R,(T,TPOS)=t
            if R not in relation:
                relation.append(R)
            if H not in vocab or T not in vocab:
        #         print(H,R,T,"0")
                continue
            else:
        #         print(H,R,T,"1")
                final_triplet.append((H,R,T))


print(len(final_triplet),len(relation))
F.save_to_file(final_triplet_file,final_triplet)
F.save_to_file(dp_relation_file,relation)

print(relation)


dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("END",time_t)