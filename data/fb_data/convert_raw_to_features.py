import os,sys

fh = open('fid_brandname_category.txt','r')
lines = fh.readlines()
fh.close()
all_unique_pids = []
for line in lines:
    arr = line.strip().split(',')
    all_unique_pids.append(arr[0].strip())

fh = open('uid_pid.txt','r')
lines = fh.readlines()
fh.close()
acts = {}
for line in lines:
    arr = line.strip().split(",")
    uid = arr[0].strip()
    pid = arr[1].strip()
    p_n = arr[2].strip()
    
    if uid not in acts:
        acts[uid] = []
    acts[uid].append(pid)

# output
print 'pid,',','.join(all_unique_pids),',label'
fh = open('pos_neg_users.txt','r')
lines = fh.readlines()
fh.close()
for line in lines:
    arr = line.strip().split(',')
    uid = arr[0].strip()
    label = arr[1].strip() 

    pids = []
    if uid in acts:
        pids = acts[uid]
   
    feature = []
    for i in range(len(all_unique_pids)):
        p = all_unique_pids[i]
        if p in pids:
            feature.append('1')
        else:
            feature.append('0')

    print uid,',',','.join(feature),',',label
