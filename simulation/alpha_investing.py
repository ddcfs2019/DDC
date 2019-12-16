import os,sys
import time

import numpy as np 

from sklearn import linear_model

import scipy.stats as stat

from skfeature.function.streaming import alpha_investing


training_fname = sys.argv[1]

fh = open(training_fname,'r')
lines = fh.readlines()
fh.close()

w = float(sys.argv[2]) # initial prob. of false positive: w0
alpha_delta = float(sys.argv[3]) # alpha_delta

# generate X and y
X = []
y = []
for i in range(len(lines)):
	line = lines[i]
	sample = []
	arr = line.strip().split(';')

	for j in range(len(arr)-1):
		elem = arr[j].strip().split(',')
		sample.append(int(elem[0].strip()))

	X.append(sample)
	y.append(int(arr[-1].strip()))

X = np.array(X)
y = np.array(y)

# generate true relevant features
truth_idx = [] # store their indicies
#true_features = []
arr = lines[0].strip().split(';')
p = len(arr) - 1
for i in range(p):
	elem = arr[i].strip().split(',')
	structure = elem[1].strip()
	if len(structure) == 1:
		truth_idx.append(i)
		#true_features.append(structure)
	else:
		temp = [int(x) for x in structure[2:].split('/')]
		if sum(temp) == 0:
			truth_idx.append(i)
			#true_features.append(structure)
truth = [0]*p
for idx in truth_idx:
	truth[idx] = 1


def tpr_fpr(truth,pred):
	# calculate true positive rate and false positive rate
	tp,fp,fn,tn = 0.0,0.0,0.0,0.0
	for i in range(len(truth)):
		t = truth[i]
		p = pred[i]
		if t == 1 and p == 1:
			tp += 1
		elif t == 1 and p == 0:
			fn += 1
		elif t == 0 and p == 1:
			fp += 1
		else:
			tn += 1
	tpr = tp / (tp + fn)
	fpr = fp / (fp + tn)

	return (tpr, fpr)


start_time = time.time()

try:
	pred_idx = alpha_investing.alpha_investing(X, y, w, alpha_delta)

	pred = [0]*p
	for idx in pred_idx:
		pred[idx] = 1

	tpr, fpr = tpr_fpr(truth,pred)
except:
	sys.exit(1)

end_time = time.time()

# output the measures
print(tpr,',',fpr,',',end_time - start_time)
