import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import random



df = pd.read_csv('res_left_deep.txt',sep='|')
numofjoins = lambda x: len(x.split('('))
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
df.sort_values(by=['numofjoins'], inplace=True,ascending=False)




# with train / test set split
#result_files = ['res_reward_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-06_09-30_trainset.txt']
result_files = ['res_reward_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-06_09-30_testset.txt']
#result_files = ['res_reward_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-07_13-23_test8400'] # all others are bad['res_reward_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-07_12-31_test8400','res_reward_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-07_12-57_test8400','res_reward_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-07_13-57_test8400','res_reward_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-07_14-28_test8400']
#result_files = ['res_reward_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-07_15-04_test15','res_reward_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-07_15-36_test15','res_reward_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-07_16-11_test15','res_reward_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-07_16-45_test15','res_reward_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-07_17-19_test15']
random.seed(7400)
#random.seed(8400)
#random.seed(15)
test_i = random.sample(range(0, len(df)), int(len(df)*0.25))
df.reset_index(inplace=True)
del df['index']
df['index'] = df.index
#print(df.head())
#df.plot.scatter(x='index', y='reward')
#plt.scatter(df['index'],df['reward'])
df_test = df.iloc[test_i]
train_i = []
for x in range(0,len(df)):
    if x not in test_i:
        train_i.append(x)
df_train = df.iloc[train_i]
del df_train['index']
del df_test['index']
df_test.reset_index(inplace=True)
#df_test["reward"].plot()
df_train.reset_index(inplace=True)
#df_train["reward"].plot()
print(df_train.describe())

for filename in result_files:
    df_test["reward"].plot()
    file = open(filename,'r')
    raw_rew=file.read().replace('\nEpisode reward ',',')
    res = []
    for l in raw_rew.split('\n'):
        x = l.split(",")
        res.append([int(x[0]),float(x[1])])

    df_res = pd.DataFrame(res,columns=["nr","reward"])
    df_res["costs"] = df_res['reward'].apply(invers_sqrt)
    print(df_res.describe())

    #df_res["costs"].plot()
    df_res["reward"].plot()
    #df_res.plot.scatter(x='nr', y='reward', c='r')
    #plt.scatter(df_res['nr'], df_res['reward'], c='r')
    print(df_res.where(df_res['reward']<=-0.35).dropna())
    plt.ylim([-0.55,0]) #1e13
    plt.show()
