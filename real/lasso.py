import os,sys
import time
import random

import numpy as np 

from sklearn import linear_model
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold


np.random.seed(2019)

training_fname = sys.argv[1] # orginial feature-based file

fh = open(training_fname,'r')
lines = fh.readlines()[1:]
fh.close()
random.shuffle(lines)

# generate X and y
X = []
y = []
for i in range(len(lines)):
	line = lines[i]
	sample = []
	arr = line.strip().split(',')
	feature = arr[1:-1]

	for j in range(len(feature)):
		sample.append(int(feature[j].strip()))

	X.append(sample)
	y.append(int(arr[-1].strip()))

X = csr_matrix(np.array(X))
y = np.array(y)


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


selection_times = []
estimation_times = []

cv = KFold(n_splits=5, shuffle=True)
fprs = []
tprs = []

for lda in np.arange(0.01,0.1,0.01): # tunable parameters
	# build the model
	ss = time.time()

	clf = linear_model.Lasso(alpha=lda,random_state=7)
	clf.fit(X,y)
	coefs = clf.coef_
	selected_indx = []
	for i in range(len(coefs)):
		if coefs[i] != 0:
			selected_indx.append(i)
	new_X = X[:,selected_indx]


	es = time.time()
	selection_times.append(es-ss)

	for threshold in np.arange(0.1,1,0.1): # thresholds

		se = time.time()

		clf = linear_model.LogisticRegression(random_state=7)

		tpr_folds = []
		fpr_folds = []

		for train,test in cv.split(new_X,y):
			try:
				clf.fit(new_X[train],y[train])
				probas_ =  clf.predict_proba(new_X[test])
				pred = np.where(probas_[:,1]>threshold,1,0)
				tpr, fpr = tpr_fpr(y[test],pred)
				tpr_folds.append(tpr)
				fpr_folds.append(fpr)
			except:
				pass

		fprs.append(np.mean(fpr_folds))
		tprs.append(np.mean(tpr_folds))
		print(lda,',',threshold,',',np.mean(fpr_folds),',',np.mean(tpr_folds))

		ee = time.time()
		estimation_times.append(ee-se)
