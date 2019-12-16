import os,sys
import time
import math

import numpy as np 

from sklearn import linear_model

import scipy
from scipy import stats

training_fname = sys.argv[1]

fh = open(training_fname,'r')
lines = fh.readlines()
fh.close()

# generate X and y
X = []
y = []
for line in lines:
	sample = []
	arr = line.strip().split(';')

	for i in range(len(arr)-1):
		elem = arr[i].strip().split(',')
		sample.append(int(elem[0].strip()))

	X.append(sample)
	y.append(int(arr[-1].strip()))

X = np.array(X)
y = np.array(y)
n, cols = X.shape

d = int(sys.argv[2]) # 10-20
#d = int(n/math.log(n))  # d= [d/log(n)] 

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


# choose top d features
start_time = time.time()

corrs = []
for c in range(cols):
	corr = stats.pearsonr(X[:,c].tolist(), y.tolist())
	corrs.append(corr)

#d = len(truth_idx)
top_d = sorted(range(len(corrs)), key=lambda i: corrs[i], reverse=True)[:d]
pred = [0]*p
for idx in top_d:
	pred[idx] = 1

# calcualte tpr and fpr
tpr, fpr = tpr_fpr(truth,pred)

end_time = time.time()

# output the measures
print(tpr,',',fpr,',',end_time - start_time)
