from __future__ import division
import re
import numpy as np

#------------PART (e) : TF-IDF ------------

n_classes = 8

#Multinomial Event Model
f = open('imdb_train_text_vec.txt', 'r')
all_lines = f.readlines()
print len(all_lines)
f.close()
for i in range(len(all_lines)):
	all_lines[i] = all_lines[i][2:-3]
	all_lines[i] = all_lines[i].split("', '")

n_docs = len(all_lines)

#Vocabulary unigrams
f = open('vocabulary.txt', 'r')
vocab_uni = f.readlines()
f.close()
vocab_uni_len = len(vocab_uni)
for i in range(vocab_uni_len):
	vocab_uni[i] = vocab_uni[i][:-1]
vocab_uni_dict = dict([ (vocab_uni[i], i) for i in range(len(vocab_uni)) ])

#Vocabulary bigrams
list_bigrams = []
for i in range(n_docs):
	for j in range(1, len(all_lines[i])):
		list_bigrams.append([all_lines[i][j-1], all_lines[i][j]])
def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item
vocab_bi = list(uniq(sorted(list_bigrams)))
vocab_bi_dict = dict([(vocab_bi[i][0]+'_'+vocab_bi[i][1], i) for i in range(len(vocab_bi)) ])
vocab_bi_len = len(vocab_bi_dict)
vocab_bi_store = np.asarray(vocab_bi)
np.savetxt('vocabulary_bigrams.csv', vocab_bi_store, delimiter=',', fmt="%s")
# vocab_bi_dict = {}
# count = 0
# for word1 in vocab_uni_dict:
# 	for word2 in vocab_uni_dict:
# 		vocab_bi_dict[word1+'_'+word2] = count
# 		count += 1
# vocab_bi_len = len(vocab_bi_dict)
print 'bigram vocabulary contructed.'

#Params for y
labels = np.genfromtxt('imdb_train_labels.txt', dtype=int)
labels_dict = {1:0, 2:1, 3:2, 4:3, 7:4, 8:5, 9:6, 10:7}
ind_to_labels = {0:1, 1:2, 2:3, 3:4, 4:7, 5:8, 6:9, 7:10}
phi_y = np.zeros(n_classes)
for key in labels_dict:
	phi_y[labels_dict[key]] = sum(labels == key)
phi_y = phi_y / len(labels)

#Learn Params for each word, label (And create index array for each document)
#Learn Unigram Priors
unigram_probs = np.ones((vocab_uni_len, n_classes), dtype=int)
size_per_class = np.zeros(n_classes, dtype=int)
X = []
list_unmatched = []
for i in range(n_docs):
	size_doc = len(all_lines[i])
	X.append(np.zeros(size_doc))
	label = labels[i]
	size_per_class[labels_dict[label]] += size_doc
	for j in range(size_doc):
		word = all_lines[i][j]
		try:
			X[i][j] = vocab_uni_dict[word]
			unigram_probs[vocab_uni_dict[word], labels_dict[label]] += 1
		except:
			list_unmatched.append(all_lines[i][j])
list_unmatched = list(set(list_unmatched))
print list_unmatched
# print X
print np.sum(size_per_class)
print sum(unigram_probs==0)
#Divide by Total Number of Words for each class
for i in range(n_classes):
	unigram_probs[:, i] = np.log(unigram_probs[:, i]) - np.log(size_per_class[i]+vocab_uni_len)
print sum(unigram_probs==0)

#Learn Bigram Priors
bigram_probs = np.ones((vocab_bi_len, n_classes), dtype=int)
size_per_class_word = np.zeros((vocab_uni_len, n_classes), dtype=int)
for i in range(n_docs):
	size_doc = len(all_lines[i])
	label = labels[i]
	for j in range(1, size_doc):
		word1 = all_lines[i][j-1]
		word2 = all_lines[i][j]
		bi_dict_idx = word1+'_'+word2
		bigram_probs[vocab_bi_dict[bi_dict_idx], labels_dict[label]] += 1
		size_per_class_word[vocab_uni_dict[word1], labels_dict[label]] += 1

print np.sum(size_per_class)
print sum(unigram_probs==0)
#Divide by Total Number of Words for word 1, and for each class (AND TAKE LOG)
for bigram in vocab_bi_dict:
	word1 = bigram.split('_')[0]
	for j in range(n_classes):
		bigram_probs[vocab_bi_dict[bigram], j] = np.log(bigram_probs[vocab_bi_dict[bigram], j]) - np.log(size_per_class_word[vocab_uni_dict[word1], j]+vocab_bi_len)
print sum(unigram_probs==0)

np.savetxt('Q1_parte_phi_words_unigrams.csv', unigram_probs, delimiter=',')
np.savetxt('Q1_parte_phi_y.csv', phi_y, delimiter=',')
np.savetxt('Q1_parte_phi_words_bigrams.csv', bigram_probs, delimiter=',')

#Calculate Training Accuracy
pred_labels_train = np.zeros(n_docs)
for i in range(n_docs):
	doc = all_lines[i]
	probs = np.log(phi_y)
	for label_idx in range(n_classes):
		probab = unigram_probs[vocab_uni_dict[doc[0]], label_idx]
		for j in range(1, len(doc)):
			bi_dict_idx = doc[j-1]+'_'+doc[j]
			word_phi = bigram_probs[vocab_bi_dict[bi_dict_idx], label_idx]
			probab += word_phi
		probs[label_idx] += probab
	idx = np.argmax(probs)
	pred_labels_train[i] = ind_to_labels[idx] 

print pred_labels_train==(labels)
acc = sum(pred_labels_train==labels)/len(labels)
print 'Training Accuracy - '+str(acc)

#Calculate Testing Accuracy
f = open('imdb_test_text_vec.txt', 'r')
all_lines = f.readlines()
f.close()
for i in range(len(all_lines)):
	all_lines[i] = all_lines[i][2:-3]
	all_lines[i] = all_lines[i].split("', '")

n_docs = len(all_lines)

labels = np.genfromtxt('imdb_test_labels.txt', dtype=int)

list_unseen_unigrams = []
list_unseen_bigrams = []
pred_labels_test = np.zeros(n_docs)
for i in range(n_docs):
	doc = all_lines[i]
	probs = np.log(phi_y)
	for label_idx in range(n_classes):
		try:
			probab = unigram_probs[vocab_uni_dict[doc[0]], label_idx]
		except:
			probab = 0
			list_unseen_unigrams.append(doc[0])
		for j in range(1, len(doc)):
			bi_dict_idx = doc[j-1]+'_'+doc[j]
			try:
				word_phi = bigram_probs[vocab_bi_dict[bi_dict_idx], label_idx]
				probab += word_phi
			except:
				list_unseen_bigrams.append(bi_dict_idx)
		probs[label_idx] += probab
	idx = np.argmax(probs)
	pred_labels_test[i] = ind_to_labels[idx] 

print len(list_unseen_unigrams)
print len(list_unseen_bigrams)
print pred_labels_test==(labels)
acc = sum(pred_labels_test==labels)/len(labels)
print 'Testing Accuracy - '+str(acc)