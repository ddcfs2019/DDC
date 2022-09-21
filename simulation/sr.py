import os,sys
import time,random

import numpy as np 

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stat
from sklearn.model_selection import KFold


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


class LogisticReg:
    """
    Wrapper Class for Logistic Regression which has the usual sklearn instance 
    in an attribute self.model, and pvalues, z scores and estimated 
    errors for each coefficient in 
    
    self.z_scores
    self.p_values
    self.sigma_estimates
    
    as well as the negative hessian of the log Likelihood (Fisher information)
    
    self.F_ij
    """
    
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        #### Get p-values for the fitted model ####
        denom = (2.0*(1.0+np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X/denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0]/sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x))*2 for x in z_scores] ### two tailed test for p-values
        
        self.z_scores = z_scores
        self.p_values = p_values
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij



training_fname = sys.argv[1] # original feature file

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


selection_times = []
estimation_times = []
percent_features = []


for w in np.arange(0.1,1,0.1): # tunable parameters
	for alpha_delta in np.arange(0.1,1,0.1): # tunable parameters
		ss = time.time()

		r,c = X.shape

		pred_idx = []
		for i in range(c):
			alpha = w / (2*(i+1))
			f = X[:,i].reshape(-1,1)

			# fit the model and get tpr and fpr
			try:
				clf = LogisticReg()
				clf.fit(f,y)

				pvalue = clf.p_values
				if pvalue[0] < alpha:
					pred_idx.append(i)
					w += alpha_delta - alpha
				else:
					w -= alpha
			except:
				pred_idx.append(i)
				pass

		random.shuffle(pred_idx)
		if len(pred_idx) == 0:
			continue

		new_X = X[:,pred_idx]
		#percent_features.append(float(len(pred_idx))/X.shape[1])

		es = time.time()
		selection_times.append(es-ss)


		for threshold in np.arange(0.1,1,0.1): # thresholds
			se = time.time()

			clf = RandomForestClassifier(random_state=0)
			probas_ = clf.fit(X,y).predict_proba(new_X)
			pred = np.where(probas_[:,1]>threshold,1,0)
			tpr, fpr = tpr_fpr(y,pred)

			ee = time.time()
			estimation_times.append(ee-se)
			print(threshold,',',w,',',alpha_delta,',',fpr,',',tpr,',',ee-se)

