import os
import threading

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn1

import functions as F
from multiprocessing import Pool


def preprocessing(filenames):
    data = ""
    sentences = []
    words = []

    # Find Sentences and save to file
    data = F.readData(filenames.corpus_name)
    import os
    if(not os.path.isfile(filenames.output_folder+'/'+filenames.sents_file_name)):
        sentences = F.getSentences(data)
        F.save_to_file(filenames.sents_file_name, sentences, filenames.output_folder)
    else:
        print("Sentences File Found")
        sentences=F.load_to_file(filenames.sents_file_name,filenames.output_folder)
    
    if(not os.path.isfile(filenames.output_folder+'/'+filenames.words_file_name))    :
        words = F.getWords(sentences)
        F.save_to_file(filenames.words_file_name, words, filenames.output_folder)
    else:
        print("Words File Found")
        words = F.load_to_file(filenames.words_file_name,filenames.output_folder)
    
    # Find Sentences and save to file
    
    print("Length of text data: ",len(data))

    # updated_words, vocab = F.getVocabulary(words, 400,filenames)
    # updated_words, vocab = F.getVocabulary(words, 300,filenames)
    # updated_words, vocab = F.getVocabulary(words, 200,filenames)
    # updated_words, vocab = F.getVocabulary(words, 100,filenames)
    # updated_words, vocab = F.getVocabulary(words, 75,filenames)
    # updated_words, vocab = F.getVocabulary(words, 50,filenames)
    # updated_words, vocab = F.getVocabulary(words, 25,filenames)
    # updated_words, vocab = F.getVocabulary(words, 20,filenames)
    # updated_words, vocab = F.getVocabulary(words, 15,filenames)
    updated_words, vocab = F.getVocabulary(words, 100,filenames)
    # updated_words, vocab = F.getVocabulary(words, 5,filenames)
    # updated_words, vocab = F.getVocabulary(words, 4,filenames)
    # updated_words, vocab = F.getVocabulary(words, 3,filenames)
    # updated_words, vocab = F.getVocabulary(words, 2,filenames)
    # updated_words, vocab = F.getVocabulary(words, 1,filenames)
    # updated_words, vocab = F.getVocabulary(words, 0,filenames)

    F.save_to_file(filenames.vocab_file, vocab, filenames.output_folder)
    F.save_to_file(filenames.updated_words_file_name, updated_words, filenames.output_folder)

    word_to_index = {}
    index_to_word = {}
    for k, v in enumerate(vocab):
        word_to_index[v] = k
        index_to_word[k] = v

    F.save_to_file(filenames.w2i_file, word_to_index, filenames.output_folder)
    F.save_to_file(filenames.i2w_file, index_to_word, filenames.output_folder)
    print(len(sentences), len(words))

def run_dp_parser(filenames, no_of_sentences, NO_OF_THREADS=5):

    os.system("mkdir -p " + filenames.output_folder + "/dp_data_pos")
    data = F.readData(filenames.corpus_name)
    print("Read Complete", len(data))

    def func(i):
        start = int(i * (len(data) / NO_OF_THREADS))
        end = int(min(((i + 1) * (len(data) / NO_OF_THREADS)), len(data)) - 1)
        print("Thread", i, start, end)
        F.genrate_triplets_full(i, data[start:end], filenames)

    t = [threading.Thread(target=func, args=(i,)) for i in range(NO_OF_THREADS)]
    t = [threading.Thread(target=func, args=(i,)) for i in range(NO_OF_THREADS)]
    for temp in t:
        temp.start()
    for temp in t:
        temp.join()
'''
vocab=""
files=""
def find_dp_triplets(filenames,NO_OF_THREADS=10):
    # Filter DP triple based on vocab
    # DP Dict to Triplet
    # vocab = F.load_to_file(filenames.vocab_file, filenames.output_folder)
    # files = os.listdir(filenames.output_folder + "/dp_data_pos")
    # os.system("mkdir -p " + filenames.output_folder + "/filtered_dp")
    # relation = []
    # final_triplet = []
    # i=0
    # for f in files:
    #     print(i)
    #     triplet_data = F.load_to_file("dp_data_pos/" + f, filenames.output_folder)
    #     # Find H R T
    #     j=0
    #     for sent in triplet_data:
    #         print(i,j)
    #         (H, HPOS), R, (T, TPOS) = sent
    #         H = H.lower()
    #         R = R.lower()
    #         T = T.lower()
    #         if R not in relation and R!="":
    #             relation.append(R)
    #         if H not in vocab or T not in vocab:
    #             #         print(H,R,T,"0")
    #             continue
    #         else:
    #             #         print(H,R,T,"1")
    #             final_triplet.append((H, R, T))
    #         j += 1
    #     # final_triplet = []
    #     i += 1
    #     if(i%20==19):
    #         print(len(final_triplet), len(relation))
    #         print(final_triplet)
    #         F.save_to_file(filenames.dp_triplet_file+"_"+str(i), final_triplet, filenames.output_folder+"/filtered_dp")
    #         final_triplet = []
    # F.save_to_file(filenames.dp_relation_file, relation, filenames.output_folder)

    # print(relation)

    # Filter DP triple based on vocab
    # DP Dict to Triplet
    global files
    files = os.listdir(filenames.output_folder + "/dp_data_pos")
    
    os.system("mkdir -p " + filenames.output_folder + "/Filtered_DP")
    os.system("mkdir -p " + filenames.output_folder + "/Relations_DP")
    t = [threading.Thread(target=filter_dp_triplets, args=(filenames,i,NO_OF_THREADS)) for i in range(NO_OF_THREADS)]
    t = [threading.Thread(target=filter_dp_triplets, args=(filenames,i,NO_OF_THREADS)) for i in range(NO_OF_THREADS)]
    for temp in t:
        temp.start()
    for temp in t:
        temp.join()

    # pool=Pool(NO_OF_THREADS)
    # results = [ pool.apply_async(filter_dp_triplets,(filenames,i,NO_OF_THREADS)) for i in range(NO_OF_THREADS)]
    # with open('output.txt','w') as f:
    #     f.write(results)
    #     f.close()

# def initial(filenames):
#     global vocab,files
#     vocab = F.load_to_file(filenames.vocab_file, filenames.output_folder)
#     files = os.listdir(filenames.output_folder + "/dp_data_pos")
    
#     os.system("mkdir -p " + filenames.output_folder + "/Filtered_DP")
#     os.system("mkdir -p " + filenames.output_folder + "/Relations_DP")

def filter_dp_triplets(filenames,i,NO_OF_THREADS):
    vocab = F.load_to_file(filenames.vocab_file, filenames.output_folder)
    # Filter DP triple based on vocab
    # DP Dict to Triplet
    start= int(i * (len(files)/NO_OF_THREADS))
    end= int((i+1) * (len(files)/NO_OF_THREADS) - 1)
    print(start,end)
    for f in files[start:end]:
        relation = []
        final_triplet = []
        triplet_data = F.load_to_file("dp_data_pos/" + f, filenames.output_folder)
        # Find H R T
        c=0
        for sent in triplet_data:
            print(c)
            (H, HPOS), R, (T, TPOS) = sent
            H = H.lower()
            R = R.lower()
            T = T.lower()
            if R not in relation and R!="":
                relation.append(R)
            if H not in vocab or T not in vocab:
                #         print(H,R,T,"0")
                continue
            else:
                #         print(H,R,T,"1")
                final_triplet.append((H, R, T))
            c += 1
        print(f)
        F.save_to_file("Filtered_DP/"+filenames.dp_triplet_file+"_"+f, final_triplet, filenames.output_folder)
        F.save_to_file("Relations_DP/"+filenames.dp_relation_file+"_"+f, relation, filenames.output_folder)
'''
def find_dp_triplets(filenames,NO_OF_THREADS=2):
    files = os.listdir(filenames.output_folder + "/dp_data_pos")
    vocab = F.load_to_file(filenames.vocab_file, filenames.output_folder)
    vocab_dict={}
    for v in vocab:
        vocab_dict[v]=""
    os.system("mkdir -p " + filenames.output_folder + "/Filtered_DP")
    os.system("mkdir -p " + filenames.output_folder + "/Relations_DP")
    
    def f(i):
        # start=NO_OF_THREADS
        # print(files[])
        start= int(i * (len(files)/NO_OF_THREADS))
        end= int((i+1) * (len(files)/NO_OF_THREADS) - 1)
        filter_dp_triplets(filenames,vocab_dict,files[start:end])

    # t = [threading.Thread(target=filter_dp_triplets, args=(filenames,i,NO_OF_THREADS)) for i in range(NO_OF_THREADS)]
    t = [threading.Thread(target=f, args=(i,)) for i in range(NO_OF_THREADS)]
    for temp in t:
        temp.start()
    for temp in t:
        temp.join()

def filter_dp_triplets(filenames,vocab_dict,files):
    # Filter DP triple based on vocab
    # DP Dict to Triplet
    print(len(files),len(vocab_dict))
    # print(start,end)
    for f in files:
        relation = []
        final_triplet = []
        triplet_data = F.load_to_file("dp_data_pos/" + f, filenames.output_folder)
        # Find H R T
        c=0
        for sent in triplet_data:
            # print(c)
            (H, HPOS), R, (T, TPOS) = sent
            H = H.lower()
            R = R.lower()
            T = T.lower()
            if R not in relation and R!="":
                relation.append(R)
            if H not in vocab_dict or T not in vocab_dict:
                #         print(H,R,T,"0")
                continue
            else:
                #         print(H,R,T,"1")
                final_triplet.append((H, R, T))
            c += 1
        print(f)
        F.save_to_file("Filtered_DP/"+filenames.dp_triplet_file+"_"+f, final_triplet, filenames.output_folder)
        F.save_to_file("Relations_DP/"+filenames.dp_relation_file+"_"+f, relation, filenames.output_folder)


def combine_dp_triplets(filenames):
    files = os.listdir(filenames.output_folder + "/Filtered_DP")
    all_triplets=[]
    c=0
    s=""
    for f in files:
        triplet_data = F.load_to_file("Filtered_DP/" + f, filenames.output_folder)
        # all_triplets += triplet_data
        for triplet in triplet_data:
            h,r,t=triplet
            s+=str(h)+"\t"+str(r)+"\t"+str(t)+"\n"
        # if c>5:
        #     break
        print(c)
        c += 1
    # F.save_to_file('all_dp_triplet',all_triplets,filenames.output_folder)    
    f = open(filenames.output_folder + '/all_dp_relations.txt','w')
    f.write(s)
    f.close()

    all_triplets=[]
    return

def combine_dp_relations(filenames):
    files = os.listdir(filenames.output_folder + "/Relations_DP")
    all_triplets=[]
    c=0
    for f in files:
        triplet_data = F.load_to_file("Relations_DP/" + f, filenames.output_folder)
        all_triplets += triplet_data
        # if c>5:
        #     break
        print(c)
        c += 1
    all_triplets=list(set(all_triplets))
    print(all_triplets)
    F.save_to_file(filenames.dp_relation_file,all_triplets,filenames.output_folder)    
    
    return

def find_co_occurences(filenames):
    # Occurence
    os.system("mkdir -p " + filenames.output_folder + "/occurences")

    data = F.load_to_file(filenames.updated_words_file_name, filenames.output_folder)
    vocab = F.load_to_file(filenames.vocab_file, filenames.output_folder)
    word_to_index = F.load_to_file(filenames.w2i_file, filenames.output_folder)
    index_to_word = F.load_to_file(filenames.i2w_file, filenames.output_folder)
    print(word_to_index)
    print(index_to_word)
    print(len(vocab), len(data))

    data_index = [word_to_index[w] for w in data]
    unknown_id = word_to_index['UKN']
    occurrence = {}
    window = 2
    print("Words:", len(data_index))
    for i in range(-window, window + 1):
        occurrence[i] = []

    for c in range(len(data_index)):
        # print(c)
        start = max(0, c - window)
        end = min(len(data_index) - 1, c + window)
        #     print(start,end)
        if data_index[c] != unknown_id:
            for j in range(start, end + 1):
                if c != j and data_index[j] != 0:
                    #                 print(j,c)
                    occurrence[j - c].append((data_index[c], data_index[j]))
        # if(c%10000000==9999999):
        if (c % 10000000 == 9999999):
            F.save_to_file("occurences/" + filenames.updated_words_file_name + str((c / 10000000) + 1), occurrence,
                           filenames.output_folder)
            for i in range(-window, window + 1):
                occurrence[i] = []

    if len(data_index) <= 10000000:
        F.save_to_file("occurences/" + filenames.updated_words_file_name + str(len(data_index)), occurrence,
                       filenames.output_folder)

    for k in occurrence:
        print(k, len(occurrence[k]))


def find_temp_co_occurences(filenames):
    # Occurence
    # os.system("mkdir -p " + filenames.output_folder + "/occurences")
    f = open(filenames.output_folder + '/occurences.txt','w')

    data = F.load_to_file(filenames.updated_words_file_name, filenames.output_folder)
    # vocab = F.load_to_file(filenames.vocab_file, filenames.output_folder)
    word_to_index = F.load_to_file(filenames.w2i_file, filenames.output_folder)
    index_to_word = F.load_to_file(filenames.i2w_file, filenames.output_folder)
    print(word_to_index)
    print(index_to_word)
    # print(len(vocab), len(data))

    data_index = [word_to_index[w] for w in data]
    unknown_id = word_to_index['UKN']
    occurrence = {}
    window = 2
    # print("Words:", len(data_index))
    # for i in range(-window, window + 1):
    #     occurrence[i] = []

    for c in range(len(data_index)):
        # print(c)
        start = max(0, c - window)
        end = min(len(data_index) - 1, c + window)
        #     print(start,end)
        if data_index[c] != unknown_id:
            for j in range(start, end + 1):
                if c != j and data_index[j] != 0:
                    #                 print(j,c)
                    # occurrence[j - c].append((data_index[c], data_index[j]))
                    f.write(str(data_index[c])+"\t"+str(data_index[j])+"\t"+str(j-c)+"\n")
        # if(c%10000000==9999999):
        # if (c % 10000000 == 9999999):
        #     F.save_to_file("occurences/" + filenames.updated_words_file_name + str((c / 10000000) + 1), occurrence,
        #                    filenames.output_folder)
        #     for i in range(-window, window + 1):
        #         occurrence[i] = []

    # if len(data_index) <= 10000000:
    #     F.save_to_file("occurences/" + filenames.updated_words_file_name + str(len(data_index)), occurrence,
    #                    filenames.output_folder)
    f.close()
    # for k in occurrence:
    #     print(k, len(occurrence[k]))

def IsStrong(syn1, syn2, w1, w2):
    # If strong 3
    # If weak 1 : w1 present in w2 definition
    # If weak 2 : w2 present in w1 definition
    w1_def = ""
    for t in syn1:
        w1_def += t.definition()
    w2_def = ""
    for t in syn2:
        w2_def += t.definition()
    flag1 = w1_def.find(w2)
    flag2 = w2_def.find(w1)
    if flag1 > -1 and flag2 > -1:
        return 3
    elif flag1 > -1:
        return 2
    elif flag2 > -1:
        return 1
    else:
        return 0

def get_relation(word1, word2):
    rel = []
    syn1 = wn1.synsets(word1)
    syn2 = wn1.synsets(word2)
    ant1 = []
    ant2 = []
    for syn in syn1:
        for l in syn.lemmas():
            if l.antonyms():
                ant1.append(l.antonyms()[0].name())
    for syn in syn2:
        for l in syn.lemmas():
            if l.antonyms():
                ant2.append(l.antonyms()[0].name())

    inter = [1 for s1 in syn1 for s2 in syn2 if s1 == s2]
    if len(inter) > 0:
        rel.append('synset')

    inter = [1 for s1 in ant1 for s2 in ant2 if s1 == s2]
    if len(inter) > 0:
        rel.append('antonym')

    is_strong = IsStrong(syn1, syn2, word1, word2)
    if is_strong == 3:
        rel.append('strong')
    elif is_strong > 0:
        rel.append('weak')



    hyponyms1 = []
    for t in syn1:
        hyponyms1 += t.hyponyms()
    hypernyms1 = []
    for t in syn1:
        hypernyms1 += t.hypernyms()
    holonyms1 = []
    for t in syn1:
        holonyms1 += t.part_holonyms()
    #     print(hyponyms1)
    inter_hypo = [1 for s1 in hyponyms1 for s2 in syn2 if s1 == s2]
    inter_hyper = [1 for s1 in hypernyms1 for s2 in syn2 if s1 == s2]
    inter_holo = [1 for s1 in holonyms1 for s2 in syn2 if s1 == s2]
    if len(inter_hypo) > 0:
        rel.append('hyponym')
    if len(inter_hyper) > 0:
        rel.append('hypernym')
    if len(inter_holo) > 0:
        rel.append('holonym')
    return rel

def find_wn_relations(filenames):
    # Wordnet Relation
    vocab = F.load_to_file(filenames.vocab_file, filenames.output_folder)
    vocab_dict={}
    for v in vocab:
        vocab_dict[v]=""
    stop = stopwords.words('english')
    stop_dict={}
    for s in stop:
        stop_dict[s]=""

    d = {}
    count = 1
    for w1 in vocab:
        print(count)
        countj = 0
        d[w1] = {}
        if w1 not in stop_dict and len(w1) > 2:
            for w2 in vocab:
                countj += 1
                if w1 != w2 and w2 not in stop and len(w2) > 2:
                    rel = get_relation(w1, w2)
                    if len(rel) > 0:
                        d[w1][w2] = rel
            print(count, countj)
            count += 1

    F.save_to_file(filenames.wordnet_triplet_file, d, filenames.output_folder)
    # a = F.load_to_file(filenames.wordnet_triplet_file, filenames.output_folder)
    # print(a)

def combine_all_triplets(filenames):
    # Positive and NUM

    vocab = F.load_to_file(filenames.vocab_file, filenames.output_folder)
    word_to_index = F.load_to_file(filenames.w2i_file, filenames.output_folder)
    index_to_word = F.load_to_file(filenames.i2w_file, filenames.output_folder)
    dp_relation = F.load_to_file(filenames.dp_relation_file, filenames.output_folder)
    # dp_triplet = F.load_to_file(filenames.dp_triplet_file, filenames.output_folder)
    # wordnet_triplet = F.load_to_file(filenames.wordnet_triplet_file, filenames.output_folder)
    # os.listdir(filenames.output_folder + "/occurences")
    # files = os.listdir(filenames.output_folder + "/occurences")
    occ = {}
    # flag = 1
    # for f in files:
    #     print(f)
    #     if flag:
    #         occ = F.load_to_file("occurences/" + f, filenames.output_folder)
    #         flag = 0
    #     else:
    #         temp_occ = F.load_to_file("occurences/" + f, filenames.output_folder)
    #         for k in occ:
    #             occ[k] += temp_occ[k]
    wordnet_relation = ['antonym','synset', 'hyponym', 'hypernym', 'holonym', 'strong', 'weak']
    # dp_relation=['advmod','amod','appos','compound','conj','fixed','flat','doeswith','list','nmod','nummod','orphan','reparandum']
    occ=[0,1,2,-1,-2]
    # wordnet_relation=F.load_to_file(filenames.wordnet_relation,filenames.output_folder)
    dp_relation = F.load_to_file(filenames.dp_relation_file, filenames.output_folder)
    print("DP rel: ",dp_relation)
    print("WN rel: ",wordnet_relation)
    print("OC rel: ",occ)

    relations = dp_relation + wordnet_relation + occ
    print(relations)
    relation_to_index = {}
    index_to_relation = {}
    for k, v in enumerate(relations):
        relation_to_index[v] = k
        index_to_relation[k] = v
    F.save_to_file(filenames.r2i_file, relation_to_index, filenames.output_folder)
    F.save_to_file(filenames.i2r_file, index_to_relation, filenames.output_folder)

    # relation_to_index = F.load_to_file(filenames.r2i_file, filenames.output_folder)
    # index_to_relation = F.load_to_file(filenames.i2r_file, filenames.output_folder)

    # print(relation_to_index)
    # print(index_to_relation)

    # dp_number_triple = []
    # dp_relation_num = [relation_to_index[r] for r in dp_relation]
    # count = 0
    # for dp_triple in dp_triplet:
    #     try:
    #         a, b, c = dp_triple
    #         a = word_to_index[a]
    #         b = relation_to_index[b]
    #         c = word_to_index[c]
    #         dp_number_triple.append((a, b, c))
    #     except:
    #         print(c)
    #         count += 1
    # len(dp_number_triple)

    # wn_number_triple = []
    # wn_relation_num = [relation_to_index[r] for r in wordnet_relation]
    # for w1 in wordnet_triplet:
    #     for w2 in wordnet_triplet[w1]:
    #         a = word_to_index[w1]
    #         b = word_to_index[w2]
    #         for c in wordnet_triplet[w1][w2]:
    #             c = relation_to_index[c]
    #             wn_number_triple.append((a, c, b))
    # len(wn_number_triple)

    # # All
    # occ_number_triple = []
    # occ_relation_num = [relation_to_index[r] for r in list(occ.keys())]
    # for r in occ:
    #     c = relation_to_index[r]
    #     for a, b in occ[r]:
    #         occ_number_triple.append((a, c, b))
    # len(occ_number_triple)

    # # without duplicates
    # occ_number_triple_without_duplicate = {}
    # occ_relation_num_without_duplicate = [relation_to_index[r] for r in list(occ.keys())]
    # for r in occ:
    #     if r < 10 and r > -10:
    #         c = relation_to_index[r]
    #         print(r, c)
    #         l = 0;
    #         for a, b in occ[r]:
    #             #         if (a,c,b) not in occ_number_triple_without_duplicate:
    #             occ_number_triple_without_duplicate[(a, c, b)] = 1
    #         print(len(occ_number_triple_without_duplicate) - l)
    # print(list(occ_number_triple_without_duplicate.keys())[:10])
    # print(len(list(occ_number_triple_without_duplicate.keys())))
    # occ_number_triple_without_dup = list(occ_number_triple_without_duplicate.keys())

    # F.save_to_file(filenames.all_relations, relations, filenames.output_folder)
    # print(len(relations))
    # print(len(wn_number_triple))
    # print(len(dp_number_triple))
    # print(len(occ_number_triple))
    # print(len(occ_number_triple_without_duplicate))

    # print(index_to_relation)

    # F.save_to_file(filenames.wn_num_file, wn_number_triple, filenames.output_folder)
    # F.save_to_file(filenames.occ_num_file, occ_number_triple, filenames.output_folder)
    # F.save_to_file(filenames.dp_num_file, dp_number_triple, filenames.output_folder)
    # F.save_to_file(filenames.occ_num_dups_file, occ_number_triple_without_dup, filenames.output_folder)

    # print(len(wn_number_triple), len(occ_number_triple), len(dp_number_triple))

    # positive_table = {}
    # total_triple = wn_number_triple + dp_number_triple + occ_number_triple_without_dup
    # for triple in total_triple:
    #     a, b, c = triple
    #     if a not in positive_table:
    #         positive_table[a] = {}
    #     if b not in positive_table[a]:
    #         positive_table[a][b] = [c]
    #     else:
    #         positive_table[a][b].append(c)

    # F.save_to_file(filenames.positive_table_file, positive_table, filenames.output_folder)