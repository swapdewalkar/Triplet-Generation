import os
import threading

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn1

import functions as F


def preprocessing(filenames):
    data = ""
    sentences = []
    words = []
    # if 's' not in F.sys.argv:
    # 	print("A")
    # Find Sentences and save to file
    data = F.readData(filenames.corpus_name)
    sentences = F.getSentences(data)
    F.save_to_file(filenames.sents_file_name, sentences, filenames.output_folder)
    # else:
    # 	print("B")
    # 	sentences=F.load_to_file(filenames.sents_file_name)

    # if 'w' not in F.sys.argv:
    print("C")
    # Find Sentences and save to file
    words = F.getWords(sentences)
    F.save_to_file(filenames.words_file_name, words, filenames.output_folder)
    # else:
    # 	print("D")
    # 	words=F.load_to_file(filenames.words_file_name)

    updated_words, vocab = F.getVocabulary(words, 400)
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

def run_dp_parser(filenames, no_of_sentences, NO_OF_THREADS=5):
    print("Swapnil")
    # sentences=F.load_to_file(filenames.sents_file_name,filenames.output_folder)
    # triplets_dict={}
    # F.count=0
    # path_to_jar = '/home/cs17mtech11004/stanford-parser-full-2018-02-27/stanford-parser.jar'
    # path_to_models_jar = '/home/cs17mtech11004/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar'
    # dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    # dependency_parser=""
    # no_of_sent=len(sentences)
    # if no_of_sentences>0:
    # 	no_of_sent=no_of_sentences
    os.system("mkdir -p " + filenames.output_folder + "/dp_data_pos")
    data = F.readData(filenames.corpus_name)
    print("Read Complete", len(data))

    def func(i):
        start = int(i * (len(data) / NO_OF_THREADS))
        end = int(min(((i + 1) * (len(data) / NO_OF_THREADS)), len(data)) - 1)
        print("Thread", i, start, end)
        F.genrate_triplets_full(i, data[start:end], filenames)

    t = [threading.Thread(target=func, args=(i,)) for i in range(NO_OF_THREADS)]
    for temp in t:
        temp.start()
    for temp in t:
        temp.join()
    # F.genrate_triplets_full(filenames)

def find_dp_triplets(filenames):
    # Filter DP triple based on vocab
    # DP Dict to Triplet
    vocab = F.load_to_file(filenames.vocab_file, filenames.output_folder)

    os.listdir(filenames.output_folder + "/dp_data_pos")
    files = os.listdir(filenames.output_folder + "/dp_data_pos")

    relation = []
    final_triplet = []
    for f in files:
        triplet_data = F.load_to_file("dp_data_pos/" + f, filenames.output_folder)
        # print(triplet_data)
        # Find H R T
        for sent in triplet_data:
            # for t in sent:
            if True:
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

    print(len(final_triplet), len(relation))
    print(final_triplet)
    F.save_to_file(filenames.dp_triplet_file, final_triplet, filenames.output_folder)
    F.save_to_file(filenames.dp_relation_file, relation, filenames.output_folder)

    print(relation)

# def find_dp_triplets(filenames):
# 	#Filter DP triple based on vocab
# 	#DP Dict to Triplet
# 	vocab=F.load_to_file(filenames.vocab_file,filenames.output_folder)


# 	os.listdir(filenames.output_folder+"/dp_data_pos")
# 	files=os.listdir(filenames.output_folder+"/dp_data_pos")

# 	relation=[]
# 	final_triplet=[]
# 	for f in files:
# 	    triplet_data=F.load_to_file("dp_data_pos/"+f,filenames.output_folder)
# 	    # print(triplet_data)
# 	    #Find H R T
# 	    for sent in triplet_data:
# 	        for t in sent:
# 	            (H,HPOS),R,(T,TPOS)=t
# 	            if R not in relation:
# 	                relation.append(R)
# 	            if H not in vocab or T not in vocab:
# 	        #         print(H,R,T,"0")
# 	                continue
# 	            else:
# 	        #         print(H,R,T,"1")
# 	                final_triplet.append((H,R,T))


# 	print(len(final_triplet),len(relation))
# 	F.save_to_file(filenames.dp_triplet_file,final_triplet,filenames.output_folder)
# 	F.save_to_file(filenames.dp_relation_file,relation,filenames.output_folder)

# 	print(relation)

# If strong 3
# If weak 1 : w1 present in w2 definition
# If weak 2 : w2 present in w1 definition

def IsStrong(syn1, syn2, w1, w2):
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
    stop = stopwords.words('english')
    d = {}
    count = 1
    for w1 in vocab:
        print(count)
        countj = 0
        d[w1] = {}
        if w1 not in stop and len(w1) > 2:
            for w2 in vocab:
                countj += 1
                if w1 != w2 and w2 not in stop and len(w2) > 2:
                    rel = get_relation(w1, w2)
                    if len(rel) > 0:
                        d[w1][w2] = rel
            print(count, countj)
            count += 1

    F.save_to_file(filenames.wordnet_triplet_file, d, filenames.output_folder)
    a = F.load_to_file(filenames.wordnet_triplet_file, filenames.output_folder)
    print(a)


def combine_all_triplets(filenames):
    # Positive and NUM

    vocab = F.load_to_file(filenames.vocab_file, filenames.output_folder)
    word_to_index = F.load_to_file(filenames.w2i_file, filenames.output_folder)
    index_to_word = F.load_to_file(filenames.i2w_file, filenames.output_folder)
    dp_relation = F.load_to_file(filenames.dp_relation_file, filenames.output_folder)
    dp_triplet = F.load_to_file(filenames.dp_triplet_file, filenames.output_folder)
    wordnet_triplet = F.load_to_file(filenames.wordnet_triplet_file, filenames.output_folder)
    os.listdir(filenames.output_folder + "/occurences")
    files = os.listdir(filenames.output_folder + "/occurences")
    occ = {}
    flag = 1
    for f in files:
        print(f)
        if flag:
            occ = F.load_to_file("occurences/" + f, filenames.output_folder)
            flag = 0
        else:
            temp_occ = F.load_to_file("occurences/" + f, filenames.output_folder)
            for k in occ:
                occ[k] += temp_occ[k]
    wordnet_relation = ['antonym','synset', 'hyponym', 'hypernym', 'holonym', 'strong', 'weak']
    # dp_relation=['advmod','amod','appos','compound','conj','fixed','flat','doeswith','list','nmod','nummod','orphan','reparandum']

    # wordnet_relation=F.load_to_file(filenames.wordnet_relation,filenames.output_folder)
    dp_relation = F.load_to_file(filenames.dp_relation_file, filenames.output_folder)
    print("DP rel: ",dp_relation)
    print("WN rel: ",wordnet_relation)
    print("OC rel: ",list(occ.keys()))

    relations = dp_relation + wordnet_relation + list(occ.keys())
    relation_to_index = {}
    index_to_relation = {}
    for k, v in enumerate(relations):
        relation_to_index[v] = k
        index_to_relation[k] = v
    F.save_to_file(filenames.r2i_file, relation_to_index, filenames.output_folder)
    F.save_to_file(filenames.i2r_file, index_to_relation, filenames.output_folder)

    relation_to_index = F.load_to_file(filenames.r2i_file, filenames.output_folder)
    index_to_relation = F.load_to_file(filenames.i2r_file, filenames.output_folder)

    print(relation_to_index)
    print(index_to_relation)

    dp_number_triple = []
    dp_relation_num = [relation_to_index[r] for r in dp_relation]
    count = 0
    for dp_triple in dp_triplet:
        try:
            a, b, c = dp_triple
            a = word_to_index[a]
            b = relation_to_index[b]
            c = word_to_index[c]
            dp_number_triple.append((a, b, c))
        except:
            print(c)
            count += 1
    len(dp_number_triple)

    wn_number_triple = []
    wn_relation_num = [relation_to_index[r] for r in wordnet_relation]
    for w1 in wordnet_triplet:
        for w2 in wordnet_triplet[w1]:
            a = word_to_index[w1]
            b = word_to_index[w2]
            for c in wordnet_triplet[w1][w2]:
                c = relation_to_index[c]
                wn_number_triple.append((a, c, b))
    len(wn_number_triple)

    # All
    occ_number_triple = []
    occ_relation_num = [relation_to_index[r] for r in list(occ.keys())]
    for r in occ:
        c = relation_to_index[r]
        for a, b in occ[r]:
            occ_number_triple.append((a, c, b))
    len(occ_number_triple)

    # without duplicates
    occ_number_triple_without_duplicate = {}
    occ_relation_num_without_duplicate = [relation_to_index[r] for r in list(occ.keys())]
    for r in occ:
        if r < 10 and r > -10:
            c = relation_to_index[r]
            print(r, c)
            l = 0;
            for a, b in occ[r]:
                #         if (a,c,b) not in occ_number_triple_without_duplicate:
                occ_number_triple_without_duplicate[(a, c, b)] = 1
            print(len(occ_number_triple_without_duplicate) - l)
    print(list(occ_number_triple_without_duplicate.keys())[:10])
    print(len(list(occ_number_triple_without_duplicate.keys())))
    occ_number_triple_without_dup = list(occ_number_triple_without_duplicate.keys())

    F.save_to_file(filenames.all_relations, relations, filenames.output_folder)
    print(len(relations))
    print(len(wn_number_triple))
    print(len(dp_number_triple))
    print(len(occ_number_triple))
    print(len(occ_number_triple_without_duplicate))

    print(index_to_relation)

    F.save_to_file(filenames.wn_num_file, wn_number_triple, filenames.output_folder)
    F.save_to_file(filenames.occ_num_file, occ_number_triple, filenames.output_folder)
    F.save_to_file(filenames.dp_num_file, dp_number_triple, filenames.output_folder)
    F.save_to_file(filenames.occ_num_dups_file, occ_number_triple_without_dup, filenames.output_folder)

    print(len(wn_number_triple), len(occ_number_triple), len(dp_number_triple))

    positive_table = {}
    total_triple = wn_number_triple + dp_number_triple + occ_number_triple_without_dup
    for triple in total_triple:
        a, b, c = triple
        if a not in positive_table:
            positive_table[a] = {}
        if b not in positive_table[a]:
            positive_table[a][b] = [c]
        else:
            positive_table[a][b].append(c)

    F.save_to_file(filenames.positive_table_file, positive_table, filenames.output_folder)
