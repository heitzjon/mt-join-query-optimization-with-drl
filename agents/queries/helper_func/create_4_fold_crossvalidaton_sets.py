import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import random
import numpy as np


df = pd.read_csv('res_left_deep.txt',sep='|')
df_rd = pd.read_csv('res_right_deep.txt',sep='|')
df_label = pd.read_csv('job_queries_simple_label2.txt',sep='|',names=['label','query'])
numofjoins = lambda x: len(x.split('('))
#print()
cost = { 'max':1.e+13, 'min':1.e+6}
sqt = lambda y : ((sqrt(y-cost['min']))/(sqrt(cost['max']-cost['min']))*-10) # SQRT
invers_sqrt  = lambda y : ((y/10*(sqrt(cost['max']-cost['min'])))**2+cost['min'])

df['numofjoins']=df['ldeepquery'].apply(numofjoins)
df['reward']=df['costs'].apply(sqt)
df['label'] = df_label['label']
#print(df.describe())


#dfkl = df
#dfkl.sort_values(by=['numofjoins'], inplace=True,ascending=False)
#dfkl['query'].to_csv("job_queries_simple_sorted_desc.txt",sep="|",index=False)
#print(df.groupby(['numofjoins']).count()['nr'])
#df.sort_values(by=['numofjoins'], inplace=True,ascending=False)

random.seed(7400)
#random.seed(8400)
#random.seed(15)
test_is=[]
test_sets = []
train_sets = []
rand_ord = random.sample(range(0, len(df)), 112)
pointer = 0
print(len(df))
for i in range(0,4):
    test_i=[]
    for j in range(0,33):
        if pointer+1<len(rand_ord):
            pointer+=1
        else:
            pointer = 0
        test_i.append(rand_ord[pointer])
    test = df.iloc[test_i]
    test_sets.append(test)
    #test["query"].to_csv("crossval/job_queries_simple_crossval_7400_"+str(i)+"_test.txt",header=False, sep="|",index=False)


    for j in test_i:
        test_is.append(j)
    train_i = []
    for x in range(0, len(df)):
        if x not in test_i:
            train_i.append(x)
    train = df.iloc[train_i]
    train_sets.append(train)
    train = train.copy()
    train.sort_values(by=['numofjoins'], inplace=True, ascending=True)

    ############### CREATE LATEX TABLE #######################3
    str = "\hline\n $"
    i_newline = 0
    i_koma = 0
    str += "[ "
    for ele in test['label'].tolist():
        if i_koma is 0:
            i_koma = 1
        else:
            str += ", "
        if i_newline == 6:
            str += "\\newline"
            i_newline = 0
        else:
            i_newline += 1
        str += ele
    str += " ]$ & \n$["
    i_newline = 0
    i_koma = 0
    for ele in train['label'].tolist():
        if i_koma is 0:
            i_koma = 1
        else:
            str += ", "
        if i_newline == 6:
            str += "\\newline"
            i_newline = 0
        else:
            i_newline += 1
        str += ele

    str += ']$\\\\\n'
    print(str)
    #train["query"].to_csv("crossval/job_queries_simple_crossval_7400_"+str(i)+"_train_sort_a.txt",header=False, sep="|",index=False)
print(len(test_is))
print(len(set(test_is)))

'''
df = pd.read_csv('res_left_deep.txt',sep='|')
df_l = pd.read_csv('job_queries_simple_label2.txt',sep='|',names=['label','query'])

list = df['query'].tolist()

#f_l['leftout']=df_l['query'].isin(list)
df_l['query2'] = df['query']

print()

for i in range(0,len(df_l)):
    print(df_l.loc[i])
#print(df_l.where(df_l['leftout']==True).dropna())
#df_l['xyz'] = df.apply(lambda x : x['label'] if x['query'] != x['query2'] else "ok")
'''