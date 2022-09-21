import os,sys
import time
import math
import random

import numpy as np 

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
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
for i in range(1,len(lines)):
	line = lines[i]
	sample = []
	arr = line.strip().split(',')
	feature = arr[1:-1]

	for j in range(len(feature)):
		sample.append(int(feature[j].strip()))

	X.append(sample)
	y.append(int(arr[-1].strip()))

X = np.array(X)
y = np.array(y)


n, cols = X.shape

# choose top d features
selection_times = []
estimation_times = []

cv = KFold(n_splits=10, shuffle=True)
fprs = []
tprs = []

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

		tpr_folds = []
		fpr_folds = []

		for train,test in cv.split(new_X,y):
			try:
				probas_ = clf.fit(csr_matrix(new_X[train]),y[train]).predict_proba(new_X[test])
				pred = np.where(probas_[:,1]>threshold,1,0)
				tpr, fpr = tpr_fpr(y[test],pred)
				tpr_folds.append(tpr)
				fpr_folds.append(fpr)
			except:
				pass

		fprs.append(np.mean(fpr_folds))
		tprs.append(np.mean(tpr_folds))

		ee = time.time()
		estimation_times.append(ee-se)
		
		print(d_percent,',',threshold,',',np.mean(fpr_folds),',',np.mean(tpr_folds),',',ee-se)
