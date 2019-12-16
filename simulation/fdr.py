import os,sys
import time
import math

import numpy as np 

from sklearn import linear_model

import scipy
from scipy import stats

training_fname = sys.argv[1]
q = float(sys.argv[2])

fh = open(training_fname,'r')
lines = fh.readlines()
fh.close()

# generate features
features = {}
y = []
for line in lines:
	arr = line.strip().split(';')
	y.append(int(arr[-1].strip()))

	for i in range(len(arr)-1):
		elem = arr[i].strip().split(',')
		k = elem[1].strip()
		v = int(elem[0].strip())
		if k not in features:
			features[k] = []
		features[k].append(v)


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
start_time = time.time()
R = ['#']
S = []
iteration = 0
while len(R) > 0:
	cur_feature = R[0]
	cur_children = get_children(cur_feature)
	if iteration == 0:
		cur_children = ['0','1','2','3','4']

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

end_time = time.time()


# generate predicted relevant features
pred_idx = []
arr = lines[0].strip().split(';')
p = len(arr) - 1
for i in range(p):
	elem = arr[i].strip().split(',')
	structure = elem[1].strip()
	if structure in S:
		pred_idx.append(i)
pred = [0]*p
for idx in pred_idx:
	pred[idx] = 1


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



# calcualte tpr and fpr
tpr, fpr = tpr_fpr(truth,pred)

# output the measures
print(tpr,',',fpr,',',end_time - start_time)
