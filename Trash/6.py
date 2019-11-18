#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import torch.nn as nn
import torch.nn.functional as fun
import functions as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os


# In[11]:
dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("Start",time_t)

#Data Loading

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
positive_table_file='Positive_Table'

word_to_index=F.load_to_file(word_to_index_file)
index_to_word=F.load_to_file(index_to_word_file)
relation_to_index=F.load_to_file(relation_to_index_file)
index_to_relation=F.load_to_file(index_to_relation_file)
wn_num=F.load_to_file(wn_num_file)
occ_num=F.load_to_file(occ_num_file)
dp_num=F.load_to_file(dp_num_file)
occ_num_dups=F.load_to_file(occ_num_dups_file)
positive_table=F.load_to_file(positive_table_file)


# In[12]:


count=0
count_r=0
for t in positive_table:
    count_r += len(positive_table[t])
    for r in positive_table[t]:
        count += len(positive_table[t][r])
print(count_r,count)


# In[9]:


print(len(wn_num),len(occ_num),len(dp_num))
print(positive_table)


# In[5]:


vocab_dim=len(index_to_word)
relation_dim=len(index_to_relation)
all_=set([x for x in range(0,vocab_dim)])
def getBatch(h):
    a,b,c=[],[],[]
    for r in positive_table[h]:
        for t in positive_table[h][r]:
            a.append(h)
            b.append(r)
            c.append(t)
    n=np.random.randint(0,vocab_dim,(len(a)))    
#     for i in range(len(a)):
#         pos=set(positive_table[h][r])
#         neg=list(all_.difference(pos))
#         seed=np.random.randint(len(neg))
# #         print(seed)
#         n.append(neg[seed])
#     print(len(a),len(b),len(c))
    return (a,b,c),(a,b,n),len(a)
print(getBatch(2))


# In[6]:


def weight_init(m): 
	if isinstance(m, nn.Linear):
		size = m.weight.size()
		fan_out = size[0] # number of rows
		fan_in = size[1] # number of columns
		variance = np.sqrt(2.0/(fan_in + fan_out))
		m.weight.data.normal_(0.0, variance)


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


# In[7]:


def train(func,name,epoch,start,one,folder,learn=0.001):
    embedding_dim=100
    vocab_dim=len(index_to_word)
    relation_dim=len(index_to_relation)
    if one==True:
        net = NetOne(embedding_dim,vocab_dim,relation_dim,func)
    else:
        net = Net(embedding_dim,vocab_dim,relation_dim,func)
    if os.path.isfile(F.folder+folder+'training_t'+name+str(start)+'.pt') and start>0:
        print("Loaded",start,one,name)
        net.load_state_dict(torch.load(F.folder+folder+'training_t'+name+str(start)+'.pt'))
    else:
        net.apply(weight_init)
    optimizer = optim.SGD(net.parameters(), lr=learn)
    MRL=nn.MarginRankingLoss(margin=1,size_average=False)
    it=0
    loss_epoch=[]
    if(start>0):
        start+=1
    for i in range(start,epoch):
        dt=F.datetime.now()
        time_t=F.datetime.strftime(dt,"%x %X")
        print(time_t)
        loss_array=[]
        for m in positive_table:
            x,x_,t=getBatch(m)
            out_p,out_n=net.forward(x,x_)
            target=Variable(torch.ones(1,t))
            loss=MRL.forward(out_p,out_n,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if it%2000==0:
                print("Batch Loss",m,loss.data.numpy()/t)
            loss_array.append(loss.data.numpy()/t)
            it += 1
        print("Epoch Mean "+str(i)+"==+++++++"+str(np.array(loss_array).mean())+"+++++++==")
        loss_epoch.append(np.array(loss_array).mean())
        F.save_to_file(folder+'loss_'+name+str(i),loss_array)
        torch.save(net.state_dict(), F.folder+folder+'training_t'+name+str(i)+'.pt')
        if(i-start>2):
            if loss_epoch[-1]<0.1 and loss_epoch[-2]<0.1:
                break
    plt.plot(range(len(loss_epoch)),loss_epoch)
    plt.show()
    F.save_to_file(folder+'loss_mean'+name,loss_epoch)
def plane(x):
    return x


# In[8]:


train(fun.sigmoid,"sigmoid_learn0.01_F",epoch=200,start=0,one=False,folder='Model/',learn=0.01)
train(fun.sigmoid,"sigmoid_learn0.01_T",epoch=200,start=0,one=True,folder='Model/',learn=0.01)
train(fun.tanh,"tanh_learn0.01_F",epoch=200,start=0,one=False,folder='Model/',learn=0.01)
train(fun.tanh,"tanh_learn0.01_T",epoch=200,start=0,one=True,folder='Model/',learn=0.01)
train(fun.sigmoid,"sigmoid_learn0.001_F",epoch=200,start=0,one=False,folder='Model/',learn=0.001)
train(fun.sigmoid,"sigmoid_learn0.001_T",epoch=200,start=0,one=True,folder='Model/',learn=0.001)
train(plane,"plane0.01_F",epoch=200,start=0,one=False,folder='Model/',learn=0.01)
train(plane,"plane0.01_T",epoch=200,start=0,one=True,folder='Model/',learn=0.01)
train(fun.leaky_relu,"Relu0.1_F",epoch=200,start=0,one=False,folder='Model/',learn=0.01)
train(fun.leaky_relu,"Relu0.1_T",epoch=200,start=0,one=True,folder='Model/',learn=0.01)
train(fun.tanh,"tanh_F",epoch=200,start=0,one=False,folder='Model/')
train(fun.tanh,"tanh_T",epoch=200,start=0,one=True,folder='Model/')
# In[50]:


# get_word_vectors(func=fun.tanh,name="tanh",epoch=4,one=False,folder='Model/')
# get_word_vectors(func=fun.sigmoid,name="sigmoid",epoch=4,one=False,folder='Model/')
# get_word_vectors(func=plane,name="Plane",epoch=4,one=False,folder='Model/')
# get_word_vectors(func=fun.tanh,name="tanh_One",epoch=4,one=True,folder='Model/')
# get_word_vectors(func=fun.sigmoid,name="sigmoid_One",epoch=4,one=True,folder='Model/')
# get_word_vectors(func=plane,name="Plane_One",epoch=4,one=True,folder='Model/')


# In[ ]:




dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("END",time_t)