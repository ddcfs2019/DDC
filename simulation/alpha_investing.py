import os,sys
import time
import random

import numpy as np 

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

import scipy.stats as stat

from skfeature.function.streaming import alpha_investing

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


np.random.seed(2019)

training_fname = sys.argv[1] # original feature file


# generate X and y
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

	for j in range(len(arr)-1):
		elem = arr[j].strip().split(',')
		sample.append(int(elem[0].strip()))

	X.append(sample)
	y.append(int(arr[-1].strip()))

X = np.array(X)
y = np.array(y)


# running
selection_times = []
estimation_times = []
percent_features = []

for w in np.arange(0.1,1,0.1): # tunable parameters
	for alpha_delta in np.arange(0.1,1,0.1): # tunable parameters

		ss = time.time()

		pred_idx = alpha_investing.alpha_investing(X, y, w, alpha_delta)
		random.shuffle(pred_idx)
		#new_X = X[:,pred_idx[:int(0.5*len(pred_idx))]]
		new_X = X[:,pred_idx]
		percent_features.append(float(len(pred_idx)+1)/X.shape[1])


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
			print(threshold,',',w,',',alpha_delta,',',fpr,',',tpr,',',ee-se)
