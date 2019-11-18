
import functions as F


# In[11]:
dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("Start",time_t)


sents_file_name='sents'
words_file_name='words'
updated_words_file_name='updated_words'
vocab_file='vocab'
w2i_file='word_to_index'
i2w_file='index_to_word'

corpus_name='../Data/reviews.txt'



data=""
sentences=[]
words=[]
if 's' not in F.sys.argv:
	print("A")
	data=F.readData(corpus_name)
	sentences=F.getSentences(data)
	F.save_to_file(sents_file_name,sentences)
else:
	print("B")
	sentences=F.load_to_file(sents_file_name)


if 'w' not in F.sys.argv:
	print("C")
	words=F.getWords(sentences)
	F.save_to_file(words_file_name,words)
else:
	print("D")
	words=F.load_to_file(words_file_name)

	
updated_words,vocab = F.getVocabulary(words,400)
F.save_to_file(vocab_file,vocab)
F.save_to_file(updated_words_file_name,updated_words)



word_to_index={}
index_to_word={}
for k,v in enumerate(vocab):
    word_to_index[v]=k
    index_to_word[k]=v
F.save_to_file(w2i_file,word_to_index)
F.save_to_file(i2w_file,index_to_word)



print(len(sentences),len(words))


dt=F.datetime.now()
time_t=F.datetime.strftime(dt,"%x %X")
print("END",time_t)