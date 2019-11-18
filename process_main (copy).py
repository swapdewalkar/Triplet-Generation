import os
import sys

import main_function as Fun

def set_output_as_file(name):
    sys.stdout = open(filenames.output_folder + "/" + name, 'w')

class FileNames:
    def __init__(self, arg, out):
        self.output_folder = "Result/" + out
        os.system("mkdir -p " + self.output_folder)
        self.corpus_name = "Data/"+arg
        self.sents_file_name = 'sents'
        self.words_file_name = 'words'
        self.lower_words_file_name = 'lower_words'
        self.updated_words_file_name = 'updated_words'
        self.vocab_file = 'vocab'
        self.w2i_file = 'word_to_index'
        self.i2w_file = 'index_to_word'
        self.r2i_file = 'relation_to_index'
        self.i2r_file = 'index_to_relation'
        self.occurrence_data_file = 'occurrence'
        self.dp_triplet_file = 'all_dp_triplet'
        self.dp_relation_file = 'dp_relation'
        self.wordnet_triplet_file = 'wordnet_relation'
        self.wn_num_file = 'wn_num'
        self.occ_num_file = 'occ_num'
        self.dp_num_file = 'dp_num'
        self.occ_num_dups_file = 'occ_num_dups'
        self.all_relations = 'all_relations'
        self.positive_table_file = 'Positive_Table'

corpus_name = sys.argv[1]
output_folder = sys.argv[2]
steps = sys.argv[3]

# corpus_name='../Data/reviews.txt'
filenames = FileNames(corpus_name, output_folder)

# python process_main.py reviews_small.txt reviews_small_output_test PDTOWC
from datetime import datetime
if 'P' in steps:
    # set_output_as_file('preprocessing.txt')
    print(str(datetime.now()))
    Fun.preprocessing(filenames)
    print(str(datetime.now()))
if 'O' in steps:
    set_output_as_file('temp_find_co_occurences.txt')
    # Fun.find_co_occurences(filenames)
    print(str(datetime.now()))
    Fun.find_temp_co_occurences(filenames)
    print(str(datetime.now()))
    
if 'W' in steps:
    set_output_as_file('find_wn_relations.txt')
    print(str(datetime.now()))
    Fun.find_wn_relations(filenames)
    print(str(datetime.now()))
    
if 'D' in steps:
    set_output_as_file('run_dp_parser.txt')
    print(str(datetime.now()))
    Fun.run_dp_parser(filenames, 0, NO_OF_THREADS=5)
    print(str(datetime.now()))
    
if 'T' in steps:
    # set_output_as_file('find_dp_triplets.txt')
    print(str(datetime.now()))
    Fun.find_dp_triplets(filenames,2)
    print(str(datetime.now()))
    
if 'A' in steps:
    set_output_as_file('combine_dp_triplets.txt')
    print(str(datetime.now()))
    Fun.combine_dp_triplets(filenames)
    print(str(datetime.now()))
    
if 'R' in steps:
    set_output_as_file('combine_dp_relations.txt')
    print(str(datetime.now()))
    Fun.combine_dp_relations(filenames)    
    print(str(datetime.now()))
    
if 'C' in steps:
    set_output_as_file('combine_all_triplets.txt')
    print(str(datetime.now()))
    Fun.combine_all_triplets(filenames)
    print(str(datetime.now()))
    


# python process_main.py reviews.txt Reviews_All_Rels O