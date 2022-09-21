import os,sys
import time
import random

import numpy as np 

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
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
	arr = line.strip().split(';')

	for j in range(len(arr)-1):
		elem = arr[j].strip().split(',')
		sample.append(int(elem[0].strip()))

	X.append(sample)
	y.append(int(arr[-1].strip()))

X = np.array(X)
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
percent_features = []

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
	#percent_features.append(float(len(selected_indx))/X.shape[1])


	es = time.time()
	selection_times.append(es-ss)
	

	for threshold in np.arange(0.1,1,0.1): # thresholds

		se = time.time()

		clf = RandomForestClassifier(random_state=7)

		clf.fit(new_X,y)
		probas_ =  clf.predict_proba(new_X)
		pred = np.where(probas_[:,1]>threshold,1,0)
		tpr, fpr = tpr_fpr(y,pred)

		ee = time.time()
		estimation_times.append(ee-se)

		print(threshold,',',lda,',',fpr,',',tpr,',',ee-se)
