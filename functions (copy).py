import pickle
from collections import Counter
from nltk import word_tokenize as wt
from nltk import sent_tokenize as st
import functions as F
import re
count=0

def clean_lower(data,options):
    if('lower' in options):
        data=data.lower()
    return data;
def clean(data,options):
    if('lower' in options):
        data=data.lower()
    if('single' in options):
        data=[d for d in data if len(d)!=1]
    if('remove_number' in options):
        data=[d for d in data if not re.match( r'.*[0-9]+.*', d)]
    if('remove_special' in options):
        data=[d for d in data if not re.match( r'.*[:;,_`=!@#$%^&*()/<>"\'\?\\\+\-\{\}\[\]\|\.]+.*', d)]
    d=Counter(data)
    v=list(d.keys())
    if('remove_less' in options):
        for k in v:
            if d[k]<less:
                del d[k]
    data=list(d.keys())
    return data

def remove_special_from_sent_data(data):
    count=0
    counto=0
    sent_new=[]
    i=0
    for sent in data:
        print(i)
        i += 1
        counto+=len(sent)
        wordsin=[ d for d in sent if not re.match( r'^[`<>=#%^*/"\'\\\{\}\[\]\|(li)]+$', d)]
        count+=len(wordsin)
        sent_new.append(wordsin)
    print("Total_word after filter single symbols",count,counto)
    return sent_new

genrated_dict={}

def wtst(i,sent_data):
    genrated_dict[i]=""
    gen=[wt(s) for s in sent_data]
    genrated_dict[i]=gen

def readData(Name):
    file=open(Name,'r')
    line=file.read()
    return line

def getSentences(data):
    sentences=st(data)
    return sentences

def getWords(sentences):
    words=[]
    for s in sentences:
        words+=wt(s) 
    return words

def getVocabulary(words,less,filenames):
    import operator
    word_lower=[]
    import os
    if(not os.path.isfile(filenames.output_folder+'/'+filenames.lower_words_file_name)):
        words_lower=[ w.lower() for w in words]  #lower
        F.save_to_file(filenames.lower_words_file_name, words_lower, filenames.output_folder)
    else:
        print("Words File Found")
        words_lower = F.load_to_file(filenames.lower_words_file_name,filenames.output_folder)

    print("Lower words count",len(words_lower))
    #remove less occuring
    d=Counter(words_lower)
    v=list(d.keys())

    #Write All word in sorted Order with their count
    f=open(filenames.output_folder+'/count_of_all_words.csv','w')
    # data_temp=sorted(d.items(),key=operator.getitem(1))
    for k in d:
        f.write(str(k)+"\t"+str(d[k])+"\n")
    f.close()    
    
    
    for k in v:
        if d[k]<less:
            del d[k]
    vocab=list(d.keys())
    print("Removing less",str(less),len(vocab))
    vocab=[w for w in vocab if not re.match( r'.*[0-9]+.*', w)]
    print("Removing Numbers",len(vocab))
    vocab=[w for w in vocab if not re.match( r'.*[:;,_`=!@#$%^&*()/<>"\'\?\\\+\-\{\}\[\]\|\.]+.*', w)]
    print("Removing Special",len(vocab))

    #Write filtered word in sorted Order with their count
    f=open(filenames.output_folder+'/count_of_filtered_words_'+str(less)+'.csv','w')
    # data_temp=sorted(d.items(),key=operator.getitem(1))
    for k in d:
        if k in vocab:
            f.write(str(k)+"\t"+str(d[k])+"\n")
    f.close()  
    updated_words=[]
    
    vocab_dict={}
    for v in vocab:
        vocab_dict[v]=""

    #Update Word to their ID for Co-Occureneces
    i=0
    for w in words_lower:
        print(i)
        if w in vocab_dict:
            updated_words.append(w)
        else:
            updated_words.append('UKN')
        i += 1
    vocab.append('UKN')
    print(len(updated_words))
    return updated_words,vocab

def getData(Name,Type,Level):
    #type
    if(Type=='text_file'):
        #readfile
        # file=open(Name)
        # line=""
        # prev="a"
        # while prev!="":
        #     prev=file.readline()
        #     line += prev + " "

        file=open(Name,'r')
        line=file.read()
        line=clean_lower(line,['lower'])    #COnvert to Lower Case
        #level
#         print(line[:100])
        if(Level=='line'):
            genrated= line;
        elif(Level=='char'):
            genrated= line.split("");
        elif(Level=='word'):
            genrated=wt(line) 
        elif(Level=='sentence_word'):
            sentences=st(line)
            genrated=[wt(s) for s in sentences]
        elif(Level=='sentence'):
            genrated=st(line)
    print("Total Sentences: ",len(genrated))
    return genrated

#Get Vocabulary from lowered corpus  
#Different Parameters can be remove_number,remove_special
def getVocab(corpus,level,ops):
    #level
    vocab_list=[]
    if(level=='word' or level=='char'):
        vocab_list=corpus;
    elif(level=='sentence_word'):
        for element in corpus:
            vocab_list+=element 
    vocab=vocab_list
    for op in ops:
#         print("H",vocab,op,"H")
        vocab=clean(vocab,op)
#         vocab=clean(vocab,'remove_number')
#         vocab=clean(vocab,'remove_special')
    vocab.append("UKN")
    print("Number of all words",len(vocab_list),"Vocabulary Size",len(vocab))
    vocab.sort()
    return vocab

#Update the Corpus given a vocabulary
def update_corpus(data,vocab,level):
    if level=='word':
        d=[]
        for w in data:
            if w in vocab:
                d.append(w)
            else:
                d.append('UKN')
    elif level=='sentence_word':
        d=[]
        for sent in data:
            t=[]
            for w in sent:
                if w in vocab:
                    t.append(w)
                else:
                    t.append('UKN')
            d.append(t)
    return d

def genrate_triplets_full(i,data,filenames):
    # data=readData(filenames.corpus_name)
    import spacy
    import math
    # print(data)
    TILE=500000
    nlp = spacy.load('en_core_web_sm')
    for j in range(math.ceil((len(data)/TILE))):
            start = int(j *TILE )
            end= int(min(((j+1) * TILE),len(data)) - 1)
            print("Thread Inside",i,j,start,end)
            doc = nlp(data[start:end])
            triplets=[]
            for token in doc:
                triplets.append(((token.head.text, token.head.pos_), token.dep_, (token.text, token.pos_)))
            # print(triplets)
            save_to_file('dp_data_pos/dp_'+str(i)+"_"+str(j),triplets,filenames.output_folder)

def save_to_file(name,data,folder):
    with open(folder+"/"+name,'wb') as f:
        pickle.dump(data,f)
        f.close()

def load_to_file(name,folder):
    with open(folder+"/"+name,'rb') as f:
        ret=pickle.load(f)
        f.close()
        return ret
