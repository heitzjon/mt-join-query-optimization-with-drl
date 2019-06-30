import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import random
from functools import reduce


df = pd.read_csv('res_left_deep.txt',sep='|')
df_ld_1_new = pd.read_csv('DP/res_left_deep_l1_new.txt',sep='|')
df_ld_2_new = pd.read_csv('DP/res_left_deep_l2_new.txt',sep='|')
df_ld_1 = pd.read_csv('DP/res_left_deep_l1.txt',sep='|')
df_ld_2 = pd.read_csv('DP/res_left_deep_l2.txt',sep='|')
df_postgres_simple = pd.read_csv('res_simple_postgres_stat.txt',sep='|')
numofjoins = lambda x: len(x.split('('))-1
#print()
cost = { 'max':1.e+13, 'min':1.e+6}
sqt = lambda y : ((sqrt(y-cost['min']))/(sqrt(cost['max']-cost['min']))*-10) # SQRT
invers_sqrt  = lambda y : ((y/10*(sqrt(cost['max']-cost['min'])))**2+cost['min'])

df['numofjoins']=df['ldeepquery'].apply(numofjoins)
df['reward']=df['costs'].apply(sqt)


print(df.describe())

#dfkl = df
#dfkl.sort_values(by=['numofjoins'], inplace=True,ascending=False)
#dfkl['query'].to_csv("job_queries_simple_sorted_desc.txt",sep="|",index=False)
print(df.groupby(['numofjoins']).count()['nr'])
#df.sort_values(by=['numofjoins'], inplace=True,ascending=False)


random.seed(7400)


#print(df_ld_2_new.describe())
#print(df_ld_2_new[df_ld_2_new['query'].duplicated()].count())


df_ld_2_new['numofjoins']=df_ld_2_new['resquery'].apply(numofjoins)
df_ld_2_new.sort_values(by=['numofjoins','costs'], inplace=True,ascending=True)


df_ld_1_new['numofjoins']=df_ld_1_new['resquery'].apply(numofjoins)
df_ld_1_new.sort_values(by=['numofjoins','costs'], inplace=True,ascending=True)


df['numofjoins']=df['ldeepquery'].apply(numofjoins)
df.sort_values(by=['numofjoins','costs'], inplace=True,ascending=True)

df_ld_2_new.reset_index(inplace=True)
df_ld_1_new.reset_index(inplace=True)
df.reset_index(inplace=True)
#del df_ld_2_new['numofjoins']
#del df_ld_2_new['index']



df['costs'].plot()
df_ld_1_new['costs'].plot()
#df_ld_2_new['costs old'].plot()
df_ld_2_new['costs'].plot()
#df_postgres_simple['costs'].plot()
#df_ld_1['costs'].plot()
#df_ld_2['costs'].plot()
#plt.ylim([-0.55,0])
plt.show()

df_box = pd.DataFrame()
df_box["DP left deep old"]=df['costs']
df_box["DP left deep 1 new"]=df_ld_1_new['costs']
df_box["DP left deep 2 new"]=df_ld_2_new['costs']
#df_box["DP left deep 1"]=df_ld_1['costs']
#df_box["DP left deep 2"]=df_ld_2['costs']


df_box.plot.box(showfliers=None)
#plt.xlabel("?")
plt.ylabel("cost value")
plt.show()














def getAllRelations(query):
    q = query.split('FROM')[1]
    q2 = q.split('WHERE')[0]
    return q2.replace(' ','').split(',')
def getRelationsOfDF(querydf):
    res = set()
    for index, row in querydf.iterrows():
        for rel in row['rel']:
            res.add(rel)
    return res
def testCrossVall(files,path):
    for file in files:
        df = pd.read_csv(path+file, sep='|',names=['query'])
        df.reset_index(inplace=True)
        #print(df.head())
        df['rel'] = df['query'].apply(getAllRelations)
        print("realtions:")
        print(len(getRelationsOfDF(df)))

        df['cond'] = df['query'].apply(getAllJoinConditions)
        print("conditions:")
        print(len(getCondsOfDF(df)))

        print("join_distr")
        numofjoins = lambda x: len(x)-1
        #df['numofjoins'] = df['ldeepquery'].apply(numofjoins)
        df['numofjoins'] = df['rel'].apply(numofjoins)
        print(df.groupby(['numofjoins']).count()['index'])

def getAllJoinConditions(query):
    q = query.split('WHERE')[1]
    return q.replace(' ','').split('AND')

def getCondsOfDF(querydf):
    res = set()
    for index, row in querydf.iterrows():
        for conds in row['cond']:
            for cond in conds.split('='):
                res.add(cond)
    return res

path = "crossval/"
files = ['job_queries_simple_crossval_7400_0_train.txt',
'job_queries_simple_crossval_7400_1_train.txt',
'job_queries_simple_crossval_7400_2_train.txt',
'job_queries_simple_crossval_7400_3_train.txt']
testCrossVall(files,path)

df_ld_1_new['cond'] = df_ld_1_new['query'].apply(getAllJoinConditions)
print(len(getCondsOfDF(df_ld_1_new)))
#print(df_ld_2_new['resquery'].head())
#print(df_ld_2_new['query'].apply(getAllRelations))
#df_ld_2_new['rel']=df_ld_2_new['query'].apply(getAllRelations)
#print(len(getRelationsOfDF(df_ld_2_new)))
#df_ld_2_new['costs old'] = df['costs']






# CREATE NEW TRAINING/TEST DATA SPLIT
#df = pd.read_csv('DP/res_left_deep_l2_new.txt',sep='|')
df = pd.read_csv('res_left_deep.txt',sep='|')
df_label = pd.read_csv('job_queries_simple_label2.txt',sep='|',names=['label','query'])
numofjoins = lambda x: len(x.split('('))-1
#df['numofjoins'] = df['resquery'].apply(numofjoins)
df['numofjoins'] = df['ldeepquery'].apply(numofjoins)
df['label'] = df_label['label']
df_s = df.loc[(df['numofjoins'] <= 6)]
df_m = df.loc[(df['numofjoins'] >= 7) & (df['numofjoins'] <= 10)]
df_l = df.loc[(df['numofjoins'] >= 11)]

print(df_s.head())

print(df_s.groupby(['numofjoins']).count()['nr'])
print(df_m.groupby(['numofjoins']).count()['nr'])
print(df_l.groupby(['numofjoins']).count()['nr'])




'''
rand_ord_s = random.sample(range(0, len(df_s)), 40)+random.sample(range(0, len(df_s)), 8)
rand_ord_m = random.sample(range(0, len(df_m)), 52)+random.sample(range(0, len(df_m)), 8)
rand_ord_l = random.sample(range(0, len(df_l)), 20)+random.sample(range(0, len(df_l)), 4)
'''
#random.seed(7400)

random.seed(8400)



rand_ord_s = list(range(0, len(df_s)))+random.sample(range(0, len(df_s)), 8)
rand_ord_m = list(range(0, len(df_m)))+random.sample(range(0, len(df_m)), 8)
rand_ord_l = list(range(0, len(df_l)))+random.sample(range(0, len(df_l)), 4)

random.shuffle(rand_ord_s)
random.shuffle(rand_ord_m)
random.shuffle(rand_ord_l)

print(rand_ord_s)
print(rand_ord_m)
print(rand_ord_l)


pointer_s = 0
pointer_m = 0
pointer_l = 0

for i in range(0,4):
    test_nr = []

    test_i=[]
    for j in range(0,12):
        if pointer_s+1<len(rand_ord_s):
            pointer_s+=1
        else:
            pointer_s = 0
        test_i.append(rand_ord_s[pointer_s])
    test_nr.append(df_s.iloc[test_i]['nr'].tolist())

    test_i=[]
    for j in range(0,15):
        if pointer_m+1<len(rand_ord_m):
            pointer_m+=1
        else:
            pointer_m = 0
        test_i.append(rand_ord_m[pointer_m])
    test_nr.append(df_m.iloc[test_i]['nr'].tolist())

    test_i = []
    for j in range(0, 6):
        if pointer_l + 1 < len(rand_ord_l):
            pointer_l += 1
        else:
            pointer_l = 0
        test_i.append(rand_ord_l[pointer_l])
    test_nr.append(df_l.iloc[test_i]['nr'].tolist())
    test_nr = reduce(lambda x, y: x + y, test_nr)
    #print(len(test_nr))
    #df_test = df.loc[df['nr'].isin(test_nr)]
    df_test = df.loc[df['nr'].reindex(test_nr)]
    #print(len(df_test))
    #print(i)
    #print(df_test.groupby(['numofjoins']).count()['nr'])
    #print([~df_test['nr'].isin(test_nr)])

    df_train = df.loc[~df['nr'].isin(test_nr)]
    ############### CREATE LATEX TABLE #######################3
    str="\hline\n $"
    i_newline = 0
    i_koma = 0
    str+="[ "
    for ele in df_test['label'].tolist():
        if i_koma is 0:
            i_koma = 1
        else:
            str+=", "
        if i_newline==6:
            str+="\\newline"
            i_newline = 0
        else:
            i_newline += 1
        str+=ele
    str+=" ]$ & \n$["
    i_newline = 0
    i_koma = 0
    for ele in df_train['label'].tolist():
        if i_koma is 0:
            i_koma = 1
        else:
            str+=", "
        if i_newline==6:
            str+="\\newline"
            i_newline = 0
        else:
            i_newline += 1
        str+=ele

    str+=']$\\\\\n'
    print(str)
    #df_test["query"].to_csv("crossval_sens/job_queries_simple_crossval_" + str(i) + "_test.txt", header=False, sep="|",index=False)
    #df_train["query"].to_csv("crossval_sens/job_queries_simple_crossval_" + str(i) + "_train.txt", header=False, sep="|",index=False)

'''
path = "crossval_sens/"
files = ['job_queries_simple_crossval_0_train.txt',
'job_queries_simple_crossval_1_train.txt',
'job_queries_simple_crossval_2_train.txt',
'job_queries_simple_crossval_3_train.txt']
testCrossVall(files,path)

'''






'''

PostgreSQL --> vergleich später



# normalize
norm = lambda y : (y-y.min())/(y.max()-y.min())
norm_df = pd.DataFrame()
norm_df["DP left deep old"]=norm(df['costs'])
norm_df["DP left deep 1 new"]=norm(df_ld_1_new['costs'])
norm_df["DP left deep 2 new"]=norm(df_ld_2_new['costs'])
norm_df['postgres_cost_s_large'] = norm(df_postgres_simple['est_cost_large'])


norm_df['numofjoins']=df_ld_2_new['resquery'].apply(numofjoins)
norm_df.sort_values(by=['numofjoins','DP left deep 2 new'], inplace=True,ascending=True)

norm_df.reset_index(inplace=True)
del norm_df['numofjoins']
del norm_df['index']
#del norm_df['level_0']


norm_df.plot()
plt.show()
'''
















'''
x = ppo["costs"]/df["costs"].mean()*100
x.plot.box(label="PPO",showfliers=False)
x =[0,1,2]
y = [100,100,100]
plt.plot(x,y,label="DP left deep mean", c='r')
y = [80,80,80]
#plt.plot(x,y,label="ReJoin")

plt.ylim([0, 900])
#plt.xlabel("?")
plt.ylabel("%")
plt.legend(loc='lower left')
plt.show()
'''
'''
df['costs'].plot()
res['costs'].plot()
df_rd['costs'].plot()
plt.show()
'''
