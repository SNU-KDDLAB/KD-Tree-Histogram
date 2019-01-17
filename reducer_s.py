#!/usr/bin/env python

from itertools import groupby
from operator import itemgetter
import sys
import random
import math


pivot_num = 5
partition_num = 32
max_dist = 50
ep = 1

def read_mapper_output(file, separator = '\t'):
    for line in file:
        yield line.rstrip().split(separator)
        
data = read_mapper_output(sys.stdin)# open('input.txt', 'r').readlines()
Q = []
O = []
pQ = []
pO = []
mQ = []
mO = []
pairs = []
real = []
for p in range(partition_num):
    pQ.append([])
    pO.append([])
    pairs.append([])
    real.append([])
    
i = 0
j = 0
na = 0
for d in data:
    #d = d.rstrip().split('\t')
    if len(d) < 2:
        continue
    if d[1] == 'Q':
        pQ[int(d[0])].append(i)
        mQ.append(list(map(int, d[2:2+pivot_num])))
        Q.append(d[2+pivot_num])
        i += 1
    else:
        pO[int(d[0])].append(j)
        mO.append(list(map(int, d[2:2+pivot_num])))
        O.append(d[2+pivot_num])
        j += 1

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
        

'''Functions for plane sweeping'''  
'''
def sort_subQ(arr,l, r, dim):
    if l >= r :
        return
    m = (l+r)/2
    sort_subQ(arr, l, m, dim)
    sort_subQ(arr, m+1, r, dim)
    l_pointer = l
    r_pointer = m+1
    temp = []
    while(l_pointer < m+1 and r_pointer < r+1):
        if mQ[arr[l_pointer][dim]<= arr[r_pointer]:
            temp.append(arr[l_pointer])
            l_pointer += 1
        else:
            
    
def sort_subO(arr,l, r, dim):
    if l >= r :
        return
    p = r
    k = l-1
    for i in range(r-l):
        if mO[arr[i+l]][dim] <= mO[arr[p]][dim]:
            temp = arr[i+l]
            arr[i+l] = arr[k+1]
            arr[k+1] = temp
            k += 1
    temp = arr[p]
    arr[p] = arr[k+1]
    arr[k+1] = temp
    sort_subO(arr, l, k, dim)
    sort_subO(arr, k+1, r, dim)
'''    
def check_ep(q, o):
    for dim1 in range(pivot_num):
        if mQ[q][dim1] + ep < mO[o][dim1]:
            return False
        if mQ[q][dim1] - ep > mO[o][dim1]:
            return False
    return True

'''Reduce step with plane sweeping'''
def plane_sweeping():
    for p in range(partition_num):
        dim = random.randrange(pivot_num)
        pQ[p] = sorted(pQ[p], key = lambda e : mQ[e][dim])#sort_subQ(pQ[p], 0, len(pQ[p])-1, dim)
        pO[p] = sorted(pO[p], key = lambda e : mO[e][dim])#sort_subO(pO[p], 0, len(pO[p])-1, dim)
        q_pointer = 0
        o_pointer = 0
        while(q_pointer < len(pQ[p]) and o_pointer < len(pO[p])):
            if mQ[pQ[p][q_pointer]][dim] < mO[pO[p][o_pointer]][dim]:
                curO = o_pointer
                while(curO < len(pO[p]) and mO[pO[p][curO]][dim] <= mQ[pQ[p][q_pointer]][dim] + ep):
                    if check_ep(pQ[p][q_pointer], pO[p][curO]):
                        pairs[p].append([pQ[p][q_pointer], pO[p][curO]])
                        if dist(Q[pQ[p][q_pointer]], O[pO[p][curO]]) <= ep:
                            real[p].append([pQ[p][q_pointer], pO[p][curO]])
                    curO += 1
                q_pointer += 1
            else:
                curQ = q_pointer
                while(curQ < len(pQ[p]) and mQ[pQ[p][curQ]][dim] <= mO[pO[p][o_pointer]][dim] + ep):
                    if check_ep(pQ[p][curQ], pO[p][o_pointer]):
                        pairs[p].append([pQ[p][curQ], pO[p][o_pointer]])
                        if dist(Q[pQ[p][curQ]], O[pO[p][o_pointer]]) <= ep:
                            real[p].append([pQ[p][curQ], pO[p][o_pointer]])
                    curQ += 1
                o_pointer += 1

plane_sweeping()
#sys.stderr.write(str(pairs))
for p in range(partition_num):
    print(len(pairs[p]))
for p in range(partition_num):
    print(str(len(pQ[p])) + ' ' +  str(len(pO[p])))
for p in range(partition_num):
    print(len(real[p]))