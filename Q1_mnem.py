from __future__ import division
import re
import numpy as np

#------------PART (a)------------

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

f = open('vocabulary.txt', 'r')
vocab = f.readlines()
f.close()
vocab_len = len(vocab)
for i in range(vocab_len):
	vocab[i] = vocab[i][:-1]
vocab_dict = dict([ (vocab[i], i) for i in range(len(vocab)) ])

#Params for y
labels = np.genfromtxt('imdb_train_labels.txt', dtype=int)
labels_dict = {1:0, 2:1, 3:2, 4:3, 7:4, 8:5, 9:6, 10:7}
ind_to_labels = {0:1, 1:2, 2:3, 3:4, 4:7, 5:8, 6:9, 7:10}
phi_y = np.zeros(n_classes)
for key in labels_dict:
	phi_y[labels_dict[key]] = sum(labels == key)
phi_y = phi_y / len(labels)

#Learn Params for each word, label (And create index array for each document)
word_param_mat = np.ones((vocab_len, n_classes), dtype=int)
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
			X[i][j] = vocab_dict[word]
			word_param_mat[vocab_dict[word], labels_dict[label]] += 1
		except:
			list_unmatched.append(all_lines[i][j])
list_unmatched = list(set(list_unmatched))
print list_unmatched
# print X
print np.sum(size_per_class)
print sum(word_param_mat==0)
#Divide by Total Number of Words for each class
for i in range(n_classes):
	word_param_mat[:, i] = np.log(word_param_mat[:, i]) - np.log(size_per_class[i]+vocab_len)
print sum(word_param_mat==0)

np.savetxt('Q1_parta_phi_words.csv', word_param_mat, delimiter=',')
np.savetxt('Q1_parta_phi_y.csv', phi_y, delimiter=',')

#Calculate Training Accuracy
pred_labels_train = np.zeros(n_docs)
for i in range(n_docs):
	doc = all_lines[i]
	probs = np.log(phi_y)
	for label_idx in range(n_classes):
		probab = 0
		for j in range(len(doc)):
			word_phi = word_param_mat[vocab_dict[doc[j]], label_idx]
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

#Calculate Testing Accuracy
list_unseen_unigrams = []
pred_labels = np.zeros(n_docs)
for i in range(n_docs):
	doc = all_lines[i]
	probs = np.log(phi_y)
	for label_idx in range(n_classes):
		probab = 0
		for j in range(len(doc)):
			try:
				word_phi = word_param_mat[vocab_dict[doc[j]], label_idx]
				probab += word_phi
			except:
				list_unseen_unigrams.append(doc[j])
		probs[label_idx] += probab
	idx = np.argmax(probs)
	pred_labels[i] = ind_to_labels[idx] 

print len(list_unseen_unigrams)
print pred_labels==labels
acc = sum(pred_labels==labels)/len(labels)
print 'Testing Accuracy - '+str(acc)

#Confusion Matrix

conf_mat = np.zeros((n_classes, n_classes), dtype=int)
for p_lbl in labels_dict:
	for a_lbl in labels_dict:
		a = (pred_labels == p_lbl)
		b = (labels == a_lbl)
		conf_mat[labels_dict[p_lbl], labels_dict[a_lbl]] = sum(a & b)

print conf_mat

print ind_to_labels[np.argmax(conf_mat.diagonal())]
conf_mat = conf_mat/np.sum(conf_mat, axis=0)
print conf_mat
print ind_to_labels[np.argmax(conf_mat.diagonal())]

#------------PART (b)------------

#Random Guessing
rand_labels = np.random.randint(0, n_classes, size=len(labels))
for i in range(len(labels)):
	rand_labels[i] = ind_to_labels[rand_labels[i]]
acc = sum(rand_labels==labels)/len(labels)
print 'Random Accuracy - '+str(acc)

#Major Guessing
maj_lbl = ind_to_labels[np.argmax(phi_y)]
maj_labels = maj_lbl*np.ones(len(labels), dtype=int)
acc = sum(maj_labels==labels)/len(labels)
print 'Major Accuracy - '+str(acc)
