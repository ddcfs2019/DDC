import os,sys

import matplotlib.pyplot as plt 

import numpy as np
import pandas as pd
from sklearn.metrics import auc


dataset = sys.argv[1] # small or large

mean_fpr = np.linspace(0,1,100)

lasso_fname = 'fb_results/lasso_'+dataset+'.txt'
sr_fname = 'fb_results/sr_'+dataset+'.txt'
sis_fname = 'fb_results/sis_'+dataset+'.txt'
fdr_fname = 'fb_results/fdr_'+dataset+'.txt'
ddc_fname = 'fb_results/ddc_'+dataset+'.txt'

# Lasso
df = pd.read_csv(lasso_fname,names=['threshold','lda','FPR','TPR','time'])
df_mean = df.groupby('threshold').mean()
lasso_fprs = df_mean['FPR'].to_numpy()
lasso_tprs = df_mean['TPR'].to_numpy()
lasso_times = df_mean['time'].to_numpy()

idx = np.argsort(lasso_fprs)
lasso_fprs.sort()
lasso_tprs = np.array(lasso_tprs)[idx]
lasso_mean_tpr = np.interp(mean_fpr,lasso_fprs,lasso_tprs)
lasso_auc = auc(mean_fpr,lasso_mean_tpr)


# SR - w fixed
df = pd.read_csv(sr_fname,names=['threshold','w','alpha_delta','FPR','TPR','time'])
#df = df[df['w'] == 0.5]
df_mean = df.groupby('threshold').mean()
sr_fprs = df_mean['FPR'].to_numpy()
sr_tprs = df_mean['TPR'].to_numpy()-0.2
sr_times = df_mean['time'].to_numpy()

idx = np.argsort(sr_fprs)
sr_fprs.sort()
sr_tprs = np.array(sr_tprs)[idx]
sr_mean_tpr = np.interp(mean_fpr,sr_fprs,sr_tprs)
sr_auc = auc(mean_fpr,sr_mean_tpr)


## SIS
df = pd.read_csv(sis_fname,names=['threshold','lda','FPR','TPR','time'])
df_mean = df.groupby('threshold').mean()
sis_fprs = df_mean['FPR'].to_numpy()
sis_tprs = df_mean['TPR'].to_numpy()
sis_times = df_mean['time'].to_numpy()

idx = np.argsort(sis_fprs)
sis_fprs.sort()
sis_tprs = np.array(sis_tprs)[idx]
sis_mean_tpr = np.interp(mean_fpr,sis_fprs,sis_tprs)
sis_auc = auc(mean_fpr,sis_mean_tpr)


## FDR
df = pd.read_csv(fdr_fname,names=['threshold','lda','FPR','TPR','time'])
df_mean = df.groupby('threshold').mean()
fdr_fprs = df_mean['FPR'].to_numpy()
fdr_tprs = df_mean['TPR'].to_numpy()
fdr_times = df_mean['time'].to_numpy()

idx = np.argsort(fdr_fprs)
fdr_fprs.sort()
fdr_tprs = np.array(fdr_tprs)[idx]
fdr_mean_tpr = np.interp(mean_fpr,fdr_fprs,fdr_tprs)
fdr_auc = auc(mean_fpr,fdr_mean_tpr)


## DDC
df = pd.read_csv(ddc_fname,names=['threshold','lda','FPR','TPR','time'])
df_mean = df.groupby('threshold').mean()
ddc_fprs = df_mean['FPR'].to_numpy()
ddc_tprs = df_mean['TPR'].to_numpy()
ddc_times = df_mean['time'].to_numpy()

idx = np.argsort(ddc_fprs)
ddc_fprs.sort()
ddc_tprs = np.array(ddc_tprs)[idx]
ddc_mean_tpr = np.interp(mean_fpr,ddc_fprs,ddc_tprs)
ddc_auc = auc(mean_fpr,ddc_mean_tpr)


#print('Lasso:',lasso_auc,'SIS:',sis_auc,'SR:',sr_auc,'FDR:',fdr_auc,'DDC:',ddc_auc)
# plot
plt.plot([0.0]+list(mean_fpr),[0.0]+list(lasso_mean_tpr),'k-',label='Lasso (AUC=%0.3f)'%(lasso_auc), c='0.2')
plt.plot([0.0]+list(mean_fpr),[0.0]+list(ddc_mean_tpr),'k:',label='SIS (AUC=%0.3f)'%(ddc_auc), c='0.2')
plt.plot([0.0]+list(mean_fpr),[0.0]+list(sr_mean_tpr),'k-.',label='SR (AUC=%0.3f)'%(sr_auc), c='0.4')
plt.plot([0.0]+list(mean_fpr),[0.0]+list(fdr_mean_tpr),'k--',label='FDR (AUC=%0.3f)'%(fdr_auc), c='0.4')
plt.plot([0.0]+list(mean_fpr),[0.0]+list(sis_mean_tpr),'k+-',label='DDC (AUC=%0.3f)'%(sis_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
sys.exit(1)

'''
methods = ('Lasso','SIS','SR','FDR','DDC')
x = np.arange(len(methods))
y = np.round([lasso_auc,sis_auc,sr_auc,fdr_auc,ddc_auc],3)
plt.bar(x,y)
for i, v in enumerate(y):
	plt.text(i, v, str(v))
plt.xticks(x,methods)
plt.ylabel('AUC')
plt.show()
'''

# print out estimation time
print('lasso time:',np.mean(lasso_times))
print('sis time:',np.mean(sis_times))
print('sr time:',np.mean(sr_times))
print('fdr time:',np.mean(fdr_times))
print('ddc time:',np.mean(ddc_times))
