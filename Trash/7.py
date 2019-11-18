import torch
import torch.nn as nn
import torch.nn.functional as fun
import functions as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os


word_to_index_file='word_to_index'
index_to_word_file='index_to_word'
word_to_index=F.load_to_file(word_to_index_file)
index_to_word=F.load_to_file(index_to_word_file)
relation_to_index_file='relation_to_index'
index_to_relation_file='index_to_relation'
relation_to_index=F.load_to_file(relation_to_index_file)
index_to_relation=F.load_to_file(index_to_relation_file)
def get_word_vectors(one,func,epoch,folder,name):
    embedding_dim=100
    vocab_dim=len(index_to_word)
    relation_dim=len(index_to_relation)
    model_file=F.folder+folder+'training_t'+name+str(epoch)+'.pt'
    print(model_file)
    # return
    if one==True:
        net = NetOne(embedding_dim,vocab_dim,relation_dim,func)
    else:
        net = Net(embedding_dim,vocab_dim,relation_dim,func)        
        
    net.load_state_dict(torch.load(model_file))
    if one==True:
        h,r=net.parameters()
    else:
        h,r,t=net.parameters()
#     with open(F.folder+folder+"word_vector/_H_"+'training_t'+name+str(epoch)+'.txt','w+') as f:
#         for i in range(len(h)):
#             f.write(str(index_to_word[i]))
#             f.write(" ")
#             eles=h[i].data.numpy()
# #             print(len(eles))
#             for ele in eles:
#                 f.write(str(ele))
#                 f.write(" ")
#             f.write("\n")
#         f.close()
#     if one==False:
#         with open(F.folder+folder+"word_vector/_T_"+'training_t'+name+str(epoch)+'.txt','w+') as f:
#             for i in range(len(t)):
#                 f.write(str(index_to_word[i]))
#                 f.write(" ")
#                 eles=t[i].data.numpy()
#     #             print(len(eles))
#                 for ele in eles:
#                     f.write(str(ele))
#                     f.write(" ")
#                 f.write("\n")
#             f.close()
	
    with open(F.folder+folder+"word_vector/_R_"+'training_t'+name+str(epoch)+'.txt','w+') as f:
        for i in range(len(r)):
            f.write(str(index_to_relation[i]))
            f.write(" ")
            eles=r[i].data.numpy()
            for ele in eles:
                f.write(str(ele))
                f.write(" ")
            f.write("\n")
        f.close()

class Net(nn.Module):
    def __init__(self,dim,vocab_dim,relation_dim,func):
        super(Net, self).__init__()
        self.act_fun=func
        self.d=dim
        self.hl = nn.Embedding(vocab_dim,dim)
        self.rl = nn.Embedding(relation_dim,dim)
        self.tl = nn.Embedding(vocab_dim,dim)
    def dist(self,h,r,t):
        res = torch.norm(h,p=2,dim=1) + torch.norm(r,p=2,dim=1) + torch.norm(t,p=2,dim=1)
        ht  = (h*t).sum(dim=1)
        rth =  (r * (t-h)).sum(dim=1)
        res = res - 2 * ( ht + rth )
        return res
    
    def forward(self, x,x_):
        h,r,t=x
        h_,r_,t_=x_
        h=torch.LongTensor(h)
        r=torch.LongTensor(r)
        t=torch.LongTensor(t)
        h_=torch.LongTensor(h_)
        r_=torch.LongTensor(r_)
        t_=torch.LongTensor(t_)
        h,r,t=Variable(h),Variable(r),Variable(t)
        h_,r_,t_=Variable(h_),Variable(r_),Variable(t_)
        
        h = self.act_fun(self.hl(h))
        r = self.act_fun(self.rl(r))
        t = self.act_fun(self.tl(t))
        h_ = self.act_fun(self.hl(h_))
        r_ = self.act_fun(self.rl(r_))
        t_ = self.act_fun(self.tl(t_))
        return self.dist(h,r,t),self.dist(h_,r_,t_)
    

class NetOne(nn.Module):
    def __init__(self,dim,vocab_dim,relation_dim,func):
        super(NetOne, self).__init__()
        self.act_fun=func
        self.d=dim
        self.hl = nn.Embedding(vocab_dim,dim)
        self.rl = nn.Embedding(relation_dim,dim)
    def dist(self,h,r,t):
        res = torch.norm(h,p=2,dim=1) + torch.norm(r,p=2,dim=1) + torch.norm(t,p=2,dim=1)
        ht  = (h*t).sum(dim=1)
        rth =  (r * (t-h)).sum(dim=1)
        res = res - 2 * ( ht + rth )
        return res
    
    def forward(self, x,x_):
        h,r,t=x
        h_,r_,t_=x_
        h=torch.LongTensor(h)
        r=torch.LongTensor(r)
        t=torch.LongTensor(t)
        h_=torch.LongTensor(h_)
        r_=torch.LongTensor(r_)
        t_=torch.LongTensor(t_)
        h,r,t=Variable(h),Variable(r),Variable(t)
        h_,r_,t_=Variable(h_),Variable(r_),Variable(t_)
        
        h = self.act_fun(self.hl(h))
        r = self.act_fun(self.rl(r))
        t = self.act_fun(self.hl(t))
        h_ = self.act_fun(self.hl(h_))
        r_ = self.act_fun(self.rl(r_))
        t_ = self.act_fun(self.hl(t_))
        return self.dist(h,r,t),self.dist(h_,r_,t_)


# get_word_vectors(func=fun.tanh,name="sigmoid_learn0.1_F",epoch=199,one=False,folder='Model/')
# get_word_vectors(func=fun.sigmoid,name="sigmoid",epoch=199,one=False,folder='Model/')
# get_word_vectors(func=plane,name="Plane",epoch=199,one=False,folder='Model/')
# get_word_vectors(func=fun.tanh,name="tanh_One",epoch=199,one=True,folder='Model/')
# get_word_vectors(func=fun.sigmoid,name="sigmoid_One",epoch=199,one=True,folder='Model/')
# get_word_vectors(func=plane,name="Plane_One",epoch=199,one=True,folder='Model/')
def plane(x):
    return x

# train(fun.sigmoid,"sigmoid_learn0.1_F",epoch=200,start=0,one=False,folder='Model/',learn=0.01)
# train(fun.tanh,"tanh_learn0.1_F",epoch=200,start=0,one=False,folder='Model/',learn=0.01)
# train(plane,"plane0.1_F",epoch=200,start=0,one=False,folder='Model/',learn=0.01)

# get_word_vectors(func=fun.sigmoid,name="sigmoid_learn0.1_F",epoch=112,one=False,folder='Model/')
# get_word_vectors(func=fun.tanh,name="tanh_learn0.1_F",epoch=199,one=False,folder='Model/')
# get_word_vectors(func=fun.leaky_relu,name="Relu0.1_F",epoch=133,one=False,folder='Model/')
# get_word_vectors(func=plane,name="plane0.1_F",epoch=149,one=False,folder='Model/')
# get_word_vectors(func=fun.sigmoid,name="sigmoid_learn0.1_T",epoch=199,one=True,folder='Model/')
# get_word_vectors(func=fun.tanh,name="tanh_learn0.1_T",epoch=199,one=True,folder='Model/')
# get_word_vectors(func=fun.leaky_relu,name="Relu0.1_T",epoch=199,one=True,folder='Model/')

# get_word_vectors(func=fun.leaky_relu,name="Relu0.1_F",epoch=199,one=False,folder='Model/')
# get_word_vectors(func=fun.leaky_relu,name="Relu0.1_T",epoch=199,one=True,folder='Model/')

# get_word_vectors(func=plane,name="plane0.1_F",epoch=199,one=False,folder='Model/')
# get_word_vectors(func=plane,name="plane0.1_T",epoch=199,one=True,folder='Model/')

# get_word_vectors(func=fun.tanh,name="tanh_learn0.1_F",epoch=199,one=False,folder='Model/')
# get_word_vectors(func=fun.tanh,name="tanh_learn0.1_T",epoch=199,one=True,folder='Model/')

# get_word_vectors(func=fun.sigmoid,name="sigmoid_learn0.1_F",epoch=199,one=False,folder='Model/')
# get_word_vectors(func=fun.sigmoid,name="sigmoid_learn0.1_T",epoch=199,one=True,folder='Model/')



get_word_vectors(func=fun.sigmoid,name="sigmoid_learn0.01_F",epoch=199,one=False,folder='Model/')
get_word_vectors(func=fun.sigmoid,name="sigmoid_learn0.01_T",epoch=199,one=True,folder='Model/')
get_word_vectors(func=fun.tanh,name="tanh_learn0.01_F",epoch=199,one=False,folder='Model/')
get_word_vectors(func=fun.tanh,name="tanh_learn0.01_T",epoch=199,one=True,folder='Model/')
get_word_vectors(func=fun.sigmoid,name="sigmoid_learn0.001_F",epoch=199,one=False,folder='Model/',)
get_word_vectors(func=fun.sigmoid,name="sigmoid_learn0.001_T",epoch=199,one=True,folder='Model/',)
get_word_vectors(func=plane,name="plane0.01_F",epoch=199,one=False,folder='Model/')
get_word_vectors(func=plane,name="plane0.01_T",epoch=199,one=True,folder='Model/')
get_word_vectors(func=fun.leaky_relu,name="Relu0.1_F",epoch=199,one=False,folder='Model/')
get_word_vectors(func=fun.leaky_relu,name="Relu0.1_T",epoch=199,one=True,folder='Model/')
get_word_vectors(func=fun.tanh,name="tanh_F",epoch=199,one=False,folder='Model/')
get_word_vectors(func=fun.tanh,name="tanh_T",epoch=199,one=True,folder='Model/')

import os
d=F.folder+"Model/word_vector/"
files=os.listdir(d)


for f in files:
    print(f)
    os.system('python filterVocab.py fullVocab.txt < '+d+f+' > '+d+'vec_'+f)
    # print('python filterVocab.py fullVocab.txt < '+d+f+' > '+d+'vec_'+f)

