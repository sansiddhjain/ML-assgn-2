from __future__ import division
import numpy as np
from svmutil import *

max_val = 255

#Read Training Data
file = np.genfromtxt('../../train.csv', delimiter=',')
X_train = file[:, :-1]
X_train = X_train/max_val
y_train = file[:, -1:]
y_train = y_train.reshape(len(y_train))

X_train = X_train.tolist()
y_train = y_train.tolist()

#Read Testing Data
file = np.genfromtxt('../../test.csv', delimiter=',')
X_test = file[:, :-1]
X_test = X_test/max_val
y_test = file[:, -1:]
y_test = y_test.reshape(len(y_test))

X_test = X_test.tolist()
y_test = y_test.tolist()

# #------------PART (c)------------

# #Train using LibSVM

# #Linear
# m = svm_train(y_train, X_train, '-t 0 -c 1')
# svm_save_model('libsvm_partc_linear.model', m)
# p_label, p_acc, p_val = svm_predict(y_test, X_test, m)

# #Gaussian
# m = svm_train(y_train, X_train, '-t 2 -g 0.05 -c 1')
# svm_save_model('libsvm_partc_gaussian.model', m)
# p_label, p_acc, p_val = svm_predict(y_test, X_test, m)

#------------PART (d)------------

# m = svm_load_model('libsvm_partc_linear.model')
# p_label, p_acc, p_val = svm_predict(y_train, X_train, m)

# m = svm_load_model('libsvm_partc_gaussian.model')
# p_label, p_acc, p_val = svm_predict(y_train, X_train, m)

# #Gaussian
# m = svm_train(y_train, X_train, '-t 2 -g 0.05 -c 10')
# svm_save_model('libsvm_partd_c10.model', m)
# p_label, p_acc, p_val = svm_predict(y_train, X_train, m)
# p_label, p_acc, p_val = svm_predict(y_test, X_test, m)

# C_arr = [1e-5, 1e-3, 1, 5, 10]
# i = 1
# for C in C_arr:
# 	svm_train(y_train, X_train, '-t 2 -g 0.05 -c '+str(C)+' -v 10')
# 	m = svm_train(y_train, X_train, '-t 2 -g 0.05 -c '+str(C)+'')
# 	svm_save_model('libsvm_partd_c_'+str(i)+'.model', m)
# 	i += 1
# 	p_label, p_acc, p_val = svm_predict(y_train, X_train, m)
# 	p_label, p_acc, p_val = svm_predict(y_test, X_test, m)

#------------PART (e)------------

m = svm_load_model('libsvm_partd_c10.model')
pred_labels, p_acc, p_val = svm_predict(y_test, X_test, m)
pred_labels = np.asarray(pred_labels)
y_test = np.asarray(y_test)
print pred_labels
n_classes = 10
conf_mat = np.zeros((n_classes, n_classes), dtype=int)
for p_lbl in range(10):
	for a_lbl in range(10):
		a = (pred_labels == p_lbl)
		b = (y_test == a_lbl)
		conf_mat[p_lbl, a_lbl] = sum(a & b)

np.savetxt('confusion_matrix.csv', conf_mat, delimiter=',', fmt="%d")
print conf_mat

print np.argmax(conf_mat.diagonal()) + 1
conf_mat = conf_mat/np.sum(conf_mat, axis=0)
print conf_mat
print np.argmax(conf_mat.diagonal()) + 1

	