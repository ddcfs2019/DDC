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

training_fname = sys.argv[1]

fh = open(training_fname,'r')
lines = fh.readlines()
fh.close()

# generate X and y
X = []
y = []
features = {}
for line in lines:
	sample = []
	arr = line.strip().split(';')
	y.append(int(arr[-1].strip()))

	for i in range(len(arr)-1):
		elem = arr[i].strip().split(',')
		k = elem[1].strip()
		v = int(elem[0].strip())
		sample.append(v)
		if k not in features:
			features[k] = []
		features[k].append(v)

	X.append(sample)

X = np.array(X)
y = np.array(y)

# calculate p-value of two-tailed t-test with n-2 df
def cal_pvalue(xi):
	n = len(xi)
	rho, pval = scipy.stats.pearsonr(xi,y)
	if rho == 1.0:
		rho = 0.9999
	tj = rho*math.sqrt(n-2)/math.sqrt(1-rho*rho)

	pj = stats.t.sf(np.abs(tj), n-2)*2

	return pj

# get all children for the current feature
def get_children(cur_feature):
	children = []
	for f in features.keys():
		if len(f) == len(cur_feature)+2 and f[:-2] == cur_feature:
			children.append(f)

	return children


# FDR
selection_times = []
estimation_times = []

cv = KFold(n_splits=10, shuffle=True)
fprs = []
tprs = []

for q in np.arange(0.1,1,0.1): # tunable parameters
	ss = time.time()

	R = ['#']
	S = []
	iteration = 0
	while len(R) > 0:
		cur_feature = R[0]
		cur_children = get_children(cur_feature)
		if iteration == 0:
			cur_children = []
			for f in list(features.keys()):
				if '/' not in f:
					cur_children.append(f)

		# doesn't have children
		if len(cur_children) == 0:
			R.remove(cur_feature)
			continue

		# has children
		m = len(cur_children)
		pvals = []
		for i in range(m):
			p = cal_pvalue(features[cur_children[i]])
			pvals.append(p)

		# sort p-values
		sorted_idx = sorted(range(len(pvals)), key=lambda i: pvals[i], reverse=False) # ascending order
		sorted_pvals = np.array(pvals)[sorted_idx].tolist()
		sorted_children = np.array(cur_children)[sorted_idx].tolist()

		# calculate r
		r = []
		for l in range(len(sorted_pvals)):
			if sorted_pvals[l] <= (l+1)*q/m:
				r.append(l+1)

		# if r is found, add features j1, j2,..., jr to R and S
		if len(r) > 0:
			for f in range(max(r)):
				R.append(sorted_children[f])
				S.append(sorted_children[f])

		R.remove(cur_feature)
		iteration += 1


	# generate predicted relevant features
	pred_idx = []
	arr = lines[0].strip().split(';')
	p = len(arr) - 1
	for i in range(p):
		elem = arr[i].strip().split(',')
		structure = elem[1].strip()
		if structure in S:
			pred_idx.append(i)

	new_X = X[:,pred_idx]


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
		
		print(q,',',threshold,',',np.mean(fpr_folds),',',np.mean(tpr_folds),',',ee-se)
