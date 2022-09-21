import os,sys
import time
import math
import random

import numpy as np 

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

import scipy
from scipy import stats

np.random.seed(2019)

training_fname = sys.argv[1] # original feature file


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


fh = open(training_fname,'r')
lines = fh.readlines()
fh.close()

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


n, cols = X.shape

# choose top d features
selection_times = []
estimation_times = []

for d_percent in np.arange(0.01,0.1,0.01): # tunable parameters: d

	ss = time.time()

	d = int(d_percent*cols)
	corrs = []
	for c in range(cols):
		corr = stats.pearsonr(X[:,c].tolist(), y.tolist())
		corrs.append(corr)

	top_d = sorted(range(len(corrs)), key=lambda i: corrs[i], reverse=True)[:d]

	new_X = X[:,top_d]

	es = time.time()
	selection_times.append(es-ss)

	for threshold in np.arange(0.1,1,0.1): # thresholds

		se = time.time()

		clf = RandomForestClassifier(random_state=0)

		probas_ = clf.fit(new_X,y).predict_proba(new_X)
		pred = np.where(probas_[:,1]>threshold,1,0)
		tpr, fpr = tpr_fpr(y,pred)

		ee = time.time()
		estimation_times.append(ee-se)

		print(threshold,',',d_percent,',',fpr,',',tpr,',',ee-se)
