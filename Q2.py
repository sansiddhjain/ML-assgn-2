from __future__ import division
import numpy as np

#------------PART (a)------------

#computes SVM decision boundary, on giving input of X, and y (with +1, -1 labels). Implements PEGASOS
def compute_svm_dec_bound(X, y, err=1e-3):
    #constants 
    m = X.shape[0]
    n = X.shape[1]
    batch_size = 100 #mini-batch size
    C = 1.0 #Regularisation param

    #init
    w = np.zeros(n)
    history_w = []
    history_w.append(w)
    n_iter = 0
    loss = 1

    while loss > err:
        rand_idxs = np.random.uniform(low=0, high=m, size=batch_size).astype(int)
        # print(rand_idxs)
        mini_batch_X = X[rand_idxs, :]
        # print(mini_batch_X)
        mini_batch_y = y[rand_idxs]
        # print(mini_batch_y)
        cond_var = np.multiply(mini_batch_y, np.dot(mini_batch_X, w))
        # print(cond_var)
        ind_arr = (cond_var < 1).astype(int)
        # print(ind_arr)
        append_to_w = np.multiply(mini_batch_X.T, np.multiply(ind_arr, mini_batch_y)).T
        append_to_w = np.multiply(C, append_to_w.sum(axis=0))
        eta_t = 1/(n_iter+1)
        w = (1-eta_t)*w + (eta_t*append_to_w)/batch_size
        # print('w'+str(w))
        history_w.append(w)
        loss = np.linalg.norm(history_w[len(history_w) - 1] - history_w[len(history_w) - 2])
        n_iter += 1
        # print('Iteration # - '+str(n_iter)+', Accuracy - '+str(accuracy(X, y, w)))

    return w

def accuracy(X, y, w):
    y_pred = 2*((np.dot(X, w) >= 0).astype(int) - 0.5)
    # y_pred = y_pred.astype(int)
    # print('y_pred'+str(y_pred))
    # print('y'+str(y))
    acc = sum(y_pred==y)/len(y)
    return acc

#------------PART (b)------------

#TRAINING

#Read data
file = np.genfromtxt('train.csv', delimiter=',')
X_master = file[:, :-1]
# X_master = X_master/255
y_ml = file[:, -1:]
y_ml = y_ml.reshape(len(y_ml))

#init
n_labels = 10
classifiers = []

#Train 45 classifiers for each pair of label
avg_train_acc = 0
for i in range(n_labels):
    for j in range(i+1, n_labels):
        ind = (y_ml == i) | (y_ml == j)
        X = X_master[ind, :]
        y = y_ml[ind]
        y[y == i] = -1
        y[y == j] = 1
        w = compute_svm_dec_bound(X, y)
        train_acc = accuracy(X, y, w)
        avg_train_acc += train_acc
        classifiers.append([i, j, w])
        print('Pair - ['+str(i)+', '+str(j)+'], Accuracy - '+str(train_acc))
avg_train_acc = avg_train_acc/45
print avg_train_acc

classifiers_model = np.zeros((len(classifiers), 786))
for i in range(len(classifiers)):
    classifier = classifiers[i]
    classifiers_model[i, 0] = classifier[0]
    classifiers_model[i, 1] = classifier[1]
    classifiers_model[i, 2:] = classifier[2]

np.savetxt('Q2_Partb_model.csv', classifiers_model, delimiter=',')

#TESTING

#Read data
file = np.genfromtxt('test.csv', delimiter=',')
X = file[:, :-1]
# X = X/255
y = file[:, -1:]
y = y.reshape(len(y))

m = X.shape[0] # No. of testing examples
y_pred = np.zeros(m, dtype=int)

for i in range(m):
    if i%500 == 0:
        print(str(i)+' iterations in testing done.')
    x = X[i, :]
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

test_acc = sum(y_pred==y)/len(y)
print('Test Accuracy - '+ str(test_acc))