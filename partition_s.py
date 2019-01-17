#!/usr/b/env python
import warnings
import math
import os
import numpy as np
import random
import queue as queue
import copy
import sys
import pickle
warnings.filterwarnings('error')

max_dist = 50
ep = 1
sample_size = 500
candidate_size = 50
pivot_num = 5
partition_num = 32
partition_num2 = 32
Q = []
O = []
dim_selection = np.random.randint(pivot_num, size= partition_num2*200)
selection = 0
smoothing_factor = 0
gamma = ep
bins = math.ceil(max_dist*gamma/ep)

with open('datasets/word/Q_11', 'rb') as f:
    Q = pickle.load(f)

with open('datasets/word/O_11', 'rb') as f:
    O = pickle.load(f)


nq = int(len(Q)/10)
no = int(len(O)/10)
Q = Q[:nq]
O = O[:no]

Q_histogram = np.zeros([pivot_num, bins],dtype=int)
O_histogram = np.zeros([pivot_num, bins],dtype=int)


def dist(s1,s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


q_sample_size = int(sample_size*nq/(nq+no)) #int(nq*sample_ratio)#
o_sample_size = sample_size - q_sample_size #int(no*sample_ratio)#
#sample_size = q_sample_size + o_sample_size

indices = np.random.choice(nq, q_sample_size, replace=False)
q_samples = Q[indices]
indices = np.random.choice(no, o_sample_size, replace=False)
o_samples = O[indices]
samples = np.concatenate((q_samples , o_samples))


def select_pivots():
    '''select outliers'''
    remain = list(range(sample_size))
    F = []
    s = random.randrange(sample_size)
    dist_matrix = -np.ones([sample_size, sample_size], dtype = int)
    temp = 0
    max = -1
    '''
    for i in range(sample_size):
        for j in range(sample_size):
            if i < j:
                dist_matrix[i][j] = dist(samples[i], samples[j])
                dist_matrix[j][i] = dist_matrix[i][j]
    '''
    #select f1
    for d in remain:
        dist_matrix[s][d] = dist(samples[s], samples[d])
        dist_matrix[d][s] = dist_matrix[s][d]
        temp2 = dist_matrix[s][d]
        if  temp2 > temp:
            max = d
            temp = temp2
    F.append(max)
    remain.remove(max)
    
    temp = 0
    max = -1
    #select f2
    for d in remain:
        if dist_matrix[F[0]][d] < 0:
            dist_matrix[F[0]][d] = dist(samples[F[0]], samples[d])
            dist_matrix[d][F[0]] = dist_matrix[F[0]][d]
        temp2 = dist_matrix[F[0]][d]
        if  temp2 > temp:
            max = d
            temp = temp2
    F.append(max)
    remain.remove(max)
    fdist = dist_matrix[F[0]][F[1]]

    #select other candidates
    while len(F) < candidate_size:
        temp = 10000000
        min = -1
        for d in remain:
            temp2 = 0
            for f in F:
                if dist_matrix[f][d] < 0:
                    dist_matrix[f][d] = dist(samples[f], samples[d])
                    dist_matrix[d][f] = dist_matrix[f][d]
                temp2 += abs(fdist-dist_matrix[f][d])
            if temp2 < temp:
                temp = temp2
                min = d
        F.append(min)
        remain.remove(min)
    del remain
    
    '''select pivots'''
    P = []
    F2 = F[:]
    qoDist = np.zeros([sample_size, sample_size])
    tempQoDist = np.zeros([sample_size, sample_size])
    curp = -1
    while len(P) < pivot_num:
        temp = 0
        max = -1
        for f in F:
            #calculate qoDist
            temp2 = 0
            for i in range(sample_size):
                for j in range(candidate_size):
                    if i < F2[j] :
                        tempQoDist[i][F2[j]] = abs(dist_matrix[f][i] - dist_matrix[f][F2[j]])
                        if tempQoDist[i][F2[j]] < qoDist[i][F2[j]]:
                            tempQoDist[i][F2[j]] = qoDist[i][F2[j]]
                        try:
                            temp2 += float(tempQoDist[i][F2[j]]/dist_matrix[i][F2[j]])
                        except Exception :
                            print(i, j, dist_matrix[i][F2[j]])
                            print(samples[i], samples[F2[j]])
                            exit()
            if temp2 > temp:
                max = f
                temp = temp2
        for i in range(sample_size):
                for j in range(candidate_size):
                    if i < F2[j] :
                        if qoDist[i][F2[j]] < abs(dist_matrix[f][i] - dist_matrix[max][F2[j]]):
                            qoDist[i][F2[j]] = abs(dist_matrix[f][i] - dist_matrix[max][F2[j]])
        P.append(max)
        F.remove(max)
    del F
    del dist_matrix
    return P
pivots = select_pivots()

'''Calulate pivot-based vetors for all data'''
mQ = []
mO = []
mS = []
mQS = []
mOS = []

for q in Q:
    temp = []
    for p in range(pivot_num):
        d = dist(q, samples[pivots[p]])
        Q_histogram[p][int(d*gamma/ep)] += 1
        temp.append(d)
    mQ.append(temp)

for o in O:
    temp = []
    for p in range(pivot_num):
        d = dist(o, samples[pivots[p]])
        O_histogram[p][int(d*gamma/ep)] += 1
        temp.append(d)
    mO.append(temp)

for s in samples:
    temp = []
    for p in range(pivot_num):
        d = dist(s, samples[pivots[p]])
        temp.append(d)
    mS.append(temp)

'''
for q_s in q_samples:
    temp = []
    for p in range(pivot_num):
        d = dist(q_s, samples[pivots[p]])
        temp.append(d)
    mQS.append(temp)

for o_s in o_samples:
    temp = []
    for p in range(pivot_num):
        d = dist(o_s, samples[pivots[p]])
        temp.append(d)
    mOS.append(temp)
'''
mQ = np.array(mQ)
mO = np.array(mO)
mS = np.array(mS)
#mQS = np.array(mQS)
#mOS = np.array(mOS)

'''Define variables for other functions'''
partitions = []
pQ = []
pO = []

for p in range(partition_num):
    pQ.append([])
    pO.append([])

'''functions and classes for KDtree partition'''
def sort_subsample(arr,l, r, dim):
    if l >= r :
        return
    p = r
    k = l-1
    for i in range(r-l):
        if mS[arr[i+l]][dim] <= mS[arr[p]][dim]:
            temp = arr[i+l]
            arr[i+l] = arr[k+1]
            arr[k+1] = temp
            k += 1
    temp = arr[p]
    arr[p] = arr[k+1]
    arr[k+1] = temp
    sort_subsample(arr, l, k, dim)
    sort_subsample(arr, k+1, r, dim)

class partition:
    def __init__(self, s_list, parent_lows, parent_highs):
        self.lows = parent_lows
        self.highs = parent_highs
        self.mlows = self.lows
        self.mhighs = self.highs
        self.s_list = s_list

class intermediate_partition:
    def __init__(self, s_list, mp, add_dim, add_low, add_high, parent_lows = None, parent_highs = None):
        self.mp = mp
        if add_dim == -1:
            self.lows = np.zeros(pivot_num)
            self.highs = np.ones(pivot_num) * max_dist
        else: 
            self.lows = parent_lows
            self.lows[add_dim] = add_low
            self.highs = parent_highs
            self.highs[add_dim] = add_high
        self.s_list = s_list
    def make_partition(self):
        return partition(copy.deepcopy(self.s_list), copy.deepcopy(self.lows), copy.deepcopy(self.highs))


'''Build KDtree and get partitions'''    
def split_intermediate_partition(S, dim):
    if S.mp == 0:
        print('Partition error!')
        exit()
    if S.mp == 1 or len(S.s_list) < 2:
        return 0, 0
    S.s_list = sorted(S.s_list, key = lambda e : mS[e][dim])#sort_subsample(S.s_list,0,len(S.s_list)-1, dim)
    mid = int((len(S.s_list)-1)/2)
    new_bound = (mS[S.s_list[mid]][dim] + mS[S.s_list[mid+1]][dim])/2.0
    S1 = intermediate_partition(copy.deepcopy(S.s_list[:mid+1]), math.ceil(S.mp/2.0), dim, 
                                S.lows[dim], new_bound, copy.deepcopy(S.lows), copy.deepcopy(S.highs))
    S2 = intermediate_partition(copy.deepcopy(S.s_list[mid+1:]), S.mp - math.ceil(S.mp/2.0), dim, 
                                new_bound, S.highs[dim], copy.deepcopy(S.lows), copy.deepcopy(S.highs))
    return S1, S2

def kdtree_partition():
    global partition_num
    global selection
    q = queue.Queue()
    q.put(intermediate_partition(list(range(sample_size)), partition_num, -1, -1, -1))
    while(q.qsize() > 0):
        S = q.get()
        #dim = random.randrange(pivot_num)
        dim = dim_selection[selection]
        selection += 1
        S1, S2 = split_intermediate_partition(S, dim)
        if(S1):
            q.put(S1)
            q.put(S2)
        else:
            partitions.append(S.make_partition())
    partition_num = len(partitions)

    
'''functions and classes for NEW KDtree partition'''
def new_kdtree_partition():
    global partition_num
    global selection
    
    histogram_product = np.zeros([pivot_num, bins], dtype = int)
    histogram_sum = np.zeros([pivot_num, bins], dtype = int)
    mids = np.zeros(pivot_num, dtype = int)
    diffs = np.zeros(pivot_num, dtype = int)
    
    for dim in range(pivot_num):
        for b in range(bins):
            histogram_sum[dim][b] += Q_histogram[dim][b]*O_histogram[dim][b]
            #for k in range(max(b - gamma + 1, 0),min(b + gamma, bins)): 
               # histogram_product[dim][b] += Q_histogram[dim][b]*O_histogram[dim][k]
        
        sum_ = np.sum(histogram_sum[dim])
        temp = 0
        for b in range(bins):
            temp += histogram_sum[dim][b]
            if temp > sum_/2:
                mids[dim] = b
                diffs[dim] = abs(2*temp - sum_)
                break
    
    
                         
    temp = diffs[0]
    min_ = 0
    for dim in range(pivot_num):
        if diffs[dim] < temp:
            min_ = dim
            temp = diffs[dim]
    bound = mids[min_]*ep/gamma + ep/gamma*0.5
    
    
    
    cut = 0
    s_list = list(range(sample_size))
    s_list = sorted(s_list, key = lambda e : mS[e][min_])
    
    for s in range(sample_size):
        if mS[s_list[s]][min_] > bound:
            cut = s
            break
            
    
    l_list = s_list[:cut]
    r_list = s_list[cut:]
    l_mp = int(partition_num*cut/sample_size)
    r_mp = partition_num - l_mp
    
   
    s_lows = np.zeros(pivot_num)
    s_highs = np.ones(pivot_num) * max_dist
    
    
    q = queue.Queue()
    q.put(intermediate_partition(l_list, l_mp, min_ , s_lows[min_], bound, copy.deepcopy(s_lows), copy.deepcopy(s_highs)))
    q.put(intermediate_partition(r_list, r_mp, min_ , bound, s_highs[min_], copy.deepcopy(s_lows), copy.deepcopy(s_highs)))
    while(q.qsize() > 0):
        S = q.get()
        #dim = random.randrange(pivot_num)
        dim = dim_selection[selection]
        selection += 1
        S1, S2 = split_intermediate_partition(S, dim)
        if(S1):
            q.put(S1)
            q.put(S2)
        else:
            partitions.append(S.make_partition())
    partition_num = len(partitions)
    
'''Generate PQis'''
def check_partitionQ(data, partition):
    for dim in range(pivot_num):
        if data[dim] >= partition.highs[dim]:
            return False
        if data[dim] < partition.lows[dim]:
            return False
    return True

def generate_PQis():
    for q in range(nq):
        temp = -1
        for p in range(partition_num):
            if check_partitionQ(mQ[q], partitions[p]):
                pQ[p].append(q)
                

'''Compute mBBs'''
def compute_mBBs():
    for p in range(partition_num):
        maxes = np.zeros(pivot_num)
        mins = np.ones(pivot_num) * max_dist
        for q in pQ[p]:
            for dim in range(pivot_num):
                if maxes[dim] < mQ[q][dim]:
                    maxes[dim] = mQ[q][dim]
                if mins[dim] > mQ[q][dim]:
                    mins[dim] = mQ[q][dim]
        partitions[p].mhighs = copy.deepcopy(maxes)
        partitions[p].mlows = copy.deepcopy(mins)

'''Generate POis'''
def check_partitionO(data, partition):
    for dim in range(pivot_num):
        if data[dim] > partition.mhighs[dim] + ep:
            return False
        if data[dim] < partition.mlows[dim] - ep:
            return False
    return True

def generate_POis():
    for o in range(no):
        for p in range(partition_num):
            if check_partitionO(mO[o], partitions[p]):
                pO[p].append(o)

def build_kd_tree():
    kdtree_partition()
    generate_PQis()
    compute_mBBs()
    generate_POis()

def new_build_kd_tree():
    new_kdtree_partition()
    generate_PQis()
    compute_mBBs()
    generate_POis()
    
def map_points():
    f = open('input.txt', 'w')
    for p in range(partition_num):
        for q_point in pQ[p]:
            f.write(str(p))
            f.write('\tQ')
            f.write('\t' + '\t'.join(map(str, mQ[q_point])))
            f.write('\t'+ Q[q_point]+ '\n')
              
        for o_point in pO[p]:
            f.write(str(p))
            f.write('\tO')
            f.write('\t' + '\t'.join(map(str, mO[o_point])))
            f.write('\t'+  O[o_point] + '\n')
    f.close()
build_kd_tree()
map_points()