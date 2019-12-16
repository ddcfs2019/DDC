import os,sys
import matplotlib.pyplot as plt

import numpy as np 

fh = open(sys.argv[1],'r')  # this is the file with the following columns: lasso_tpr, lasso_fpr, lass_time, sis_tpr, sis_fpr, sis_time, sr_tpr, sr_fpr, sr_time, fdr_tpr, fdr_fpr, fdr_time, ddc_tpr, ddc_fpr, ddc_time 
lines = fh.readlines()
fh.close()

tprs_lasso,tprs_sis,tprs_sr_w0,tprs_sr_alpha,tprs_fdr,tprs_ddc = [],[],[],[],[],[]
fprs_lasso,fprs_sis,fprs_sr_w0,fprs_sr_alpha,fprs_fdr,fprs_ddc = [],[],[],[],[],[]
times_lasso,times_sis,times_sr_w0,times_sr_alpha,times_fdr,times_ddc = [],[],[],[],[],[]
for i in range(1,len(lines)):
	arr = lines[i].strip().split(',')
	tprs_lasso.append(float(arr[0].strip()))
	fprs_lasso.append(float(arr[1].strip()))
	times_lasso.append(float(arr[2].strip()))

	tprs_sis.append(float(arr[3].strip()))
	fprs_sis.append(float(arr[4].strip()))
	times_sis.append(float(arr[5].strip()))

	tprs_sr_w0.append(float(arr[6].strip()))
	fprs_sr_w0.append(float(arr[7].strip()))
	times_sr_w0.append(float(arr[8].strip()))

	tprs_sr_alpha.append(float(arr[9].strip()))
	fprs_sr_alpha.append(float(arr[10].strip()))
	times_sr_alpha.append(float(arr[11].strip()))

	tprs_fdr.append(float(arr[12].strip()))
	fprs_fdr.append(float(arr[13].strip()))
	times_fdr.append(float(arr[14].strip()))

	tprs_ddc.append(float(arr[15].strip()))
	fprs_ddc.append(float(arr[16].strip()))
	times_ddc.append(float(arr[17].strip()))

'''
# plot FPR-TPR
plt.plot(fprs_lasso,tprs_lasso,'-k*',label='Lasso')
plt.plot(fprs_sis,tprs_sis,'-b+',label='SIS')
plt.plot(fprs_sr_w0,tprs_sr_w0,'-go',label=r'SR - $W_0$ fixed')
plt.plot(fprs_sr_alpha,tprs_sr_alpha,'-cx',label=r'SR - $\alpha_\Delta$ fixed')
plt.plot(fprs_fdr,tprs_fdr,'-m^',label='FDR')
plt.plot(fprs_ddc,tprs_ddc,'-rs',label='DDC')

plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

'''

# plot time
t_lasso = np.mean(times_lasso)
t_sis = np.mean(times_sis)
t_sr_w0 = np.mean(times_sr_w0)
t_sr_alpha = np.mean(times_sr_alpha)
t_fdr = np.mean(times_fdr)
t_ddc = np.mean(times_ddc)

methods = ('Lasso','SIS',r'SR - $W_0$ fixed',r'SR - $\alpha_\Delta$ fixed','FDR','DDC')
x = np.arange(len(methods))
y = [t_lasso,t_sis,t_sr_w0,t_sr_alpha,t_fdr,t_ddc]
plt.bar(x,y)
plt.xticks(x,methods)
plt.ylabel('Time(seconds)')
plt.show()
