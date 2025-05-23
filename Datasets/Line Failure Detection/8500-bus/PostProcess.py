# -*- coding: utf-8 -*-
"""
Created on Wed May  7 12:03:04 2025

@author: dro210000
"""
import pickle
import random


f = open('LineFailures_8500_Normal.pkl','rb')
Batch1=pickle.load(f)
f.close() 

g = open('LineFailures_8500_Attack1.pkl','rb')
Batch2=pickle.load(g)
g.close() 

h = open('LineFailures_8500_Attack2.pkl','rb')
Batch3=pickle.load(h)
h.close() 

i = open('LineFailures_8500_Attack3.pkl','rb')
Batch4=pickle.load(i)
i.close() 

j = open('LineFailures_8500_Attack4.pkl','rb')
Batch5=pickle.load(j)
j.close() 

k = open('LineFailures_8500_Attack5.pkl','rb')
Batch6=pickle.load(k)
k.close() 

Total = Batch1 + Batch2 + Batch3 + Batch4 + Batch5 + Batch6
random.shuffle(Total)

x = open('LineFailures_8500.pkl', 'wb')
pickle.dump(Total,x)
x.close()