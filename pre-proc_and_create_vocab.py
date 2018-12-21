from __future__ import division
import re
import numpy as np

n_classes = 8

f = open('imdb_train_text_stem.txt', 'r')

all_lines = f.readlines()
save_the_lines = []

#Pre-processing
for i in range(len(all_lines)):
	all_lines[i] = all_lines[i].lower()	
	all_lines[i] = re.sub(r'<br />', '', all_lines[i])
	all_lines[i] = re.sub(r'\d/10', '', all_lines[i])
	all_lines[i] = re.sub(r'\n', '', all_lines[i])
	all_lines[i] = re.sub(r'[^\sa-zA-Z0-9]+', '', all_lines[i])
	# all_lines[i] = re.sub(r'[^a-zA-Z0-9]+', ' ', all_lines[i])
	all_lines[i] = re.sub(r'\s+', ' ', all_lines[i]) #Not that important
	all_lines[i] = all_lines[i].split()
	save_the_lines.append(list(set(all_lines[i])))
	save_the_lines[i].sort() 

# # Save the modified training text (For Naive Bayes)
# for i in range(len(save_the_lines)):
# 	save_the_lines[i] = str(save_the_lines[i])
# 	save_the_lines[i] = save_the_lines[i][:] + '\n'
# with open('imdb_train_text_vec_un1.txt', 'w') as ff:
# 	ff.writelines(save_the_lines)
# ff.close()

# # Save the modified training text (For Multinomial Event Model)
# for i in range(len(all_lines)):
# 	all_lines[i] = str(all_lines[i])
# 	all_lines[i] = all_lines[i][:] + '\n'
# with open('imdb_train_text_vec1.txt', 'w') as ff:
# 	ff.writelines(all_lines)
# ff.close()
# print('Pre-processing Training done.')

# f = open('imdb_test_text_stem.txt', 'r')

# all_lines = f.readlines()
# save_the_lines = []

# # #Pre-processing
# for i in range(len(all_lines)):
# 	all_lines[i] = all_lines[i].lower()	
# 	all_lines[i] = re.sub(r'<br />', '', all_lines[i])
# 	all_lines[i] = re.sub(r'\d/10', '', all_lines[i])
# 	all_lines[i] = re.sub(r'\n', '', all_lines[i])
# 	all_lines[i] = re.sub(r'[^\sa-zA-Z0-9]+', '', all_lines[i])
# 	# all_lines[i] = re.sub(r'[^a-zA-Z0-9]+', ' ', all_lines[i])
# 	all_lines[i] = re.sub(r'\s+', ' ', all_lines[i]) #Not that important
# 	all_lines[i] = all_lines[i].split()
# 	save_the_lines.append(list(set(all_lines[i])))
# 	save_the_lines[i].sort() 

# # Save the modified training text (For Naive Bayes)
# for i in range(len(save_the_lines)):
# 	save_the_lines[i] = str(save_the_lines[i])
# 	save_the_lines[i] = save_the_lines[i][:] + '\n'
# with open('imdb_test_text_vec_un1.txt', 'w') as ff:
# 	ff.writelines(save_the_lines)
# ff.close()

# # Save the modified training text (For Multinomial Event Model)
# for i in range(len(all_lines)):
# 	all_lines[i] = str(all_lines[i])
# 	all_lines[i] = all_lines[i][:] + '\n'
# with open('imdb_test_text_vec1.txt', 'w') as ff:
# 	ff.writelines(all_lines)
# ff.close()
# print('Pre-processing Testing done.')


#Vocab Creation
vocab = all_lines[0]
print vocab

for i in range(1, len(all_lines)):
	lit = save_the_lines[i]
	vocab = vocab + lit

vocab = list(set(vocab))
vocab.sort()

#Save the vocabulary
for i in range(len(vocab)):
	vocab[i] = vocab[i][:] + '\n'
with open('vocabulary1.txt', 'w') as ff:
	ff.writelines(vocab)
ff.close()
print('Vocab saved.')