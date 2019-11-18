import os
import sys

import main_function as Fun


def set_output_as_file(name):
    sys.stdout = open(filenames.output_folder + "/" + name, 'w')


class FileNames:
    def __init__(self, arg, out):
        self.output_folder = "Result/" + out
        os.system("mkdir -p " + self.output_folder)
        self.corpus_name = arg
        self.sents_file_name = 'sents'
        self.words_file_name = 'words'
        self.updated_words_file_name = 'updated_words'
        self.vocab_file = 'vocab'
        self.w2i_file = 'word_to_index'
        self.i2w_file = 'index_to_word'
        self.r2i_file = 'relation_to_index'
        self.i2r_file = 'index_to_relation'
        self.occurrence_data_file = 'occurrence'
        self.dp_triplet_file = 'dp_triplets'
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
# corpus_name='../Data/reviews.txt'
filenames = FileNames(corpus_name, output_folder)
# set_output_as_file('preprocessing.txt')
# Fun.preprocessing(filenames)
set_output_as_file('run_dp_parser.txt')
Fun.run_dp_parser(filenames, 0, NO_OF_THREADS=15)
set_output_as_file('find_co_occurences.txt')
Fun.find_co_occurences(filenames)
set_output_as_file('find_dp_triplets.txt')
Fun.find_dp_triplets(filenames)
set_output_as_file('find_wn_relations.txt')
Fun.find_wn_relations(filenames)
set_output_as_file('combine_all_triplets.txt')
Fun.combine_all_triplets(filenames)
