import time
import numpy as np

def corpus_word_array(file_name):
	with open(file_name) as corpus_file:
		corpus_word_array = corpus_file.read().split()
	return corpus_word_array
	
def dictionary(file_name, min_frequency):
	word_array = corpus_word_array(file_name)
	print("There are " + str(len(word_array)) + " words in corpus.")
	dictionary = {}
	for word in word_array:
		if word in dictionary:
			dictionary[word] += 1
		else:
			dictionary[word] = 1
			# print("Added " + word + " to dictionary.")
	print("Size of dictionary: " + str(len(dictionary)))
	top_words = []
	covered = 0
	for word in dictionary:
		if dictionary[word] >= min_frequency:
			top_words.append(word)
			covered += dictionary[word]
	print("Number of words occuring more than minimum freuqecy: " + str(len(top_words)))
	print("Percentage covered by top words: " + str(1.0*covered/len(word_array)*100))
	return word_array, top_words

def one_hot_vector(i, l):
	one_hot_vector = np.zeros(l)
	one_hot_vector[i] = 1
	return one_hot_vector

def code_dictionary(dictionary):
	codes = {}
	for word, i in zip(dictionary, range(len(dictionary))):
		codes[word] = i	
	return codes

def coded_array(word_array, codes):
	coded_corpus = []
	for word in word_array:
		if word in codes:
			coded_corpus.append(codes[word])
		else:
			coded_corpus.append(codes['unknown'])
	return np.array(coded_corpus)

# SCRIPT
# start_time = time.time()
# print(coded_corpus(corpus_word_array('text8'), code_dictionary(dictionary('text8', 70))))
# print("Time taken in seconds: " + str(time.time()-start_time))