
import numpy as np 
import re
import nltk
from svmutil import *
from stopword_stem import getStemmedDocument
import sys

q_no = int(sys.argv[1])
m_no = int(sys.argv[2])
input_file_name = sys.argv[3]
output_file_name = sys.argv[4]

# labels = np.genfromtxt('imdb_test_labels.txt', dtype=int)

if q_no == 1:
	f = open(input_file_name, 'r')
	all_lines = f.readlines()
	n_docs = len(all_lines)
	n_classes = 8
	
	labels_dict = {1:0, 2:1, 3:2, 4:3, 7:4, 8:5, 9:6, 10:7}
	ind_to_labels = {0:1, 1:2, 2:3, 3:4, 4:7, 5:8, 6:9, 7:10}

	#Default Part(a) implementation
	if (m_no == 1):
		#Pre-processing 
		for i in range(len(all_lines)):
			all_lines[i] = all_lines[i].lower()	
			all_lines[i] = re.sub(r'<br />', '', all_lines[i])
			all_lines[i] = re.sub(r'\d/10', '', all_lines[i])
			all_lines[i] = re.sub(r'\n', '', all_lines[i])
			all_lines[i] = re.sub(r'[^\sa-zA-Z0-9]+', '', all_lines[i])
			all_lines[i] = re.sub(r'\s+', ' ', all_lines[i])
			all_lines[i] = all_lines[i].split()
		
		f = open('vocabulary.txt', 'r')
		vocab = f.readlines()
		f.close()
		vocab_len = len(vocab)
		for i in range(vocab_len):
			vocab[i] = vocab[i][:-1]
		vocab_dict = dict([ (vocab[i], i) for i in range(len(vocab)) ])

		word_param_mat = np.genfromtxt('Q1_parta_phi_words.csv', delimiter=',')
		phi_y = np.genfromtxt('Q1_parta_phi_y.csv', delimiter=',')

		#Calculate Testing Accuracy
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
						continue
				probs[label_idx] += probab
			idx = np.argmax(probs)
			pred_labels[i] = ind_to_labels[idx] 

		pred_labels = pred_labels.astype(int)
		np.savetxt(output_file_name, pred_labels, fmt="%s")
		# with open(output_file_name, 'w') as f:
		# 	f.writelines(pred_labels)

		# acc = sum(pred_labels==labels)/len(labels)
		# print('Testing Accuracy - '+str(acc))

	#Part(d) implementation - with stemming
	if (m_no == 2):
		stemmed_file_name = input_file_name.split('.')
		stemmed_file_name = stemmed_file_name[0]+'_stemmed.'+stemmed_file_name[1]
		getStemmedDocument(input_file_name, stemmed_file_name)
		
		f = open(stemmed_file_name, 'r')
		all_lines = f.readlines()
		n_docs = len(all_lines)

		for i in range(len(all_lines)):
			all_lines[i] = all_lines[i].split()
		
		f = open('vocabulary_stem.txt', 'r')
		vocab = f.readlines()
		f.close()
		vocab_len = len(vocab)
		for i in range(vocab_len):
			vocab[i] = vocab[i][:-1]
		vocab_dict = dict([ (vocab[i], i) for i in range(len(vocab)) ])

		word_param_mat = np.genfromtxt('Q1_partd_phi_words.csv', delimiter=',')
		phi_y = np.genfromtxt('Q1_partd_phi_y.csv', delimiter=',')

		#Calculate Testing Accuracy
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
						continue
				probs[label_idx] += probab
			idx = np.argmax(probs)
			pred_labels[i] = ind_to_labels[idx] 

		pred_labels = pred_labels.astype(int)	
		np.savetxt(output_file_name, pred_labels, fmt="%d")
		# with open(output_file_name, 'w') as f:
		# 	f.writelines(pred_labels)

		# acc = sum(pred_labels==labels)/len(labels)
		# print('Testing Accuracy - '+str(acc))

	#Bigrams Implementation
	if (m_no == 3):	
		#Pre-processing 
		for i in range(len(all_lines)):
			all_lines[i] = all_lines[i].lower()	
			all_lines[i] = re.sub(r'<br />', '', all_lines[i])
			all_lines[i] = re.sub(r'\d/10', '', all_lines[i])
			all_lines[i] = re.sub(r'\n', '', all_lines[i])
			all_lines[i] = re.sub(r'[^\sa-zA-Z0-9]+', '', all_lines[i])
			all_lines[i] = re.sub(r'\s+', ' ', all_lines[i])
			all_lines[i] = all_lines[i].split()
			
		f = open('vocabulary.txt', 'r')
		vocab = f.readlines()
		f.close()
		vocab_len = len(vocab)
		for i in range(vocab_len):
			vocab[i] = vocab[i][:-1]
		vocab_dict = dict([ (vocab[i], i) for i in range(len(vocab)) ])

		vocab_bi = np.genfromtxt('vocabulary_bigrams.csv', delimiter=',', dtype=str)
		vocab_bi = vocab_bi.tolist()
		vocab_bi_dict = dict([(vocab_bi[i][0]+'_'+vocab_bi[i][1], i) for i in range(len(vocab_bi)) ])
		
		unigram_probs = np.genfromtxt('Q1_parte_phi_words_unigrams.csv', delimiter=',')
		phi_y = np.genfromtxt('Q1_parta_phi_y.csv', delimiter=',')
		bigram_probs = np.genfromtxt('Q1_parte_phi_words_bigrams.csv', delimiter=',')

		pred_labels_test = np.zeros(n_docs)
		for i in range(n_docs):
			doc = all_lines[i]
			probs = np.log(phi_y)
			for label_idx in range(n_classes):
				try:
					probab = unigram_probs[vocab_uni_dict[doc[0]], label_idx]
				except:
					probab = 0
				for j in range(1, len(doc)):
					bi_dict_idx = doc[j-1]+'_'+doc[j]
					try:
						word_phi = bigram_probs[vocab_bi_dict[bi_dict_idx], label_idx]
						probab += word_phi
					except:
						continue
				probs[label_idx] += probab
			idx = np.argmax(probs)
			pred_labels_test[i] = ind_to_labels[idx] 

		# print pred_labels_test==(labels)
		# acc = sum(pred_labels_test==labels)/len(labels)
		# print 'Testing Accuracy - '+str(acc)
		pred_labels_test = pred_labels_test.astype(int)
		np.savetxt(output_file_name, pred_labels_test, fmt="%d")
		# with open(output_file_name, 'w') as f:
		# 		f.writelines(pred_labels_test)

		# acc = sum(pred_labels_test==labels)/len(labels)
		# print('Testing Accuracy - '+str(acc))

if q_no == 2:
	#Read Testing Data
	max_val = 255
	n_labels = 10
	file = np.genfromtxt(input_file_name, delimiter=',')
	X_test = file[:, :]
	# X_test = file[:, :-1]
	# print(X_test.shape)
	X_test = X_test/max_val
	# y_test = file[:, -1:]
	# y_test = y_test.reshape(len(y_test))

	if m_no == 1:
		classifiers_model = np.genfromtxt('Q2_Partb_model.csv', delimiter=',')
		classifiers = []
		for i in range(classifiers_model.shape[0]):
			neg_label = int(classifiers_model[i, 0])
			pos_label = int(classifiers_model[i, 1])
			w = classifiers_model[i, 2:]
			classifiers.append([neg_label, pos_label, w])
		
		m = X_test.shape[0] # No. of testing examples
		y_pred = np.zeros(m, dtype=int)

		for i in range(m):
		    x = X_test[i, :]
		    label_counter = np.zeros(n_labels, dtype=int)
		    for classifier in classifiers:
		        neg_label = classifier[0]
		        pos_label = classifier[1]
		        w = classifier[2]
		        if np.dot(x, w) >= 0:
		            label_counter[pos_label] += 1
		        else:
		            label_counter[neg_label] += 1
		    max_val = np.amax(label_counter)
		    max_idxs = np.where(label_counter == max_val)
		    y_pred[i] = np.amax(max_idxs)

		# print(sum(y_pred==y_test)/len(y_test))
		y_pred = y_pred.astype(int)
		np.savetxt(output_file_name, y_pred, fmt="%d")
		# with open(output_file_name, 'w') as f:
		# 	f.writelines(y_pred)

	if m_no == 2:
		X_test = X_test.tolist()
		y_test = np.zeros(X_test.shape[0], dtype=int)

		m = svm_load_model('libsvm_partc_linear.model')
		p_label, p_acc, p_val = svm_predict(y_test, X_test, m)
		p_label = [str(int(i)) for i in p_label]
		with open(output_file_name, 'w') as f:
			f.writelines(p_label)

	if m_no == 3:
		X_test = X_test.tolist()
		y_test = np.zeros(X_test.shape[0], dtype=int)

		m = svm_load_model('libsvm_partd_c10.model')
		p_label, p_acc, p_val = svm_predict(y_test, X_test, m)
		p_label = [str(int(i)) for i in p_label]
		with open(output_file_name, 'w') as f:
			f.writelines(p_label)



