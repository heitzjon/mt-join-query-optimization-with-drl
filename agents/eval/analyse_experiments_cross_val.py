import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import random


def combineCrossValFiles(path, result_files,test_sets,test_is):
    i = 0
    xy = 0
    df_res_list = []
    for filename in result_files:
        test_sets[i]["reward"].plot()
        file = open(path + filename, 'r')
        raw_rew = file.read().replace('\nEpisode reward ', ',')
        res = []
        for l in raw_rew.split('\n'):
            x = l.split(",")
            res.append([int(x[0]), float(x[1])])

        df_res = pd.DataFrame(res, columns=["ex_nr", "reward"])
        df_res["costs"] = df_res['reward'].apply(invers_sqrt)
        df_res["nr"] = test_is[i]
        # print(df_res.describe())

        # df_res["costs"].plot()
        df_res["reward"].plot()
        # df_res.plot.scatter(x='nr', y='reward', c='r')
        # plt.scatter(df_res['nr'], df_res['reward'], c='r')
        print(df_res.where(df_res['reward'] <= -0.35).dropna())
        plt.ylim([-0.55, 0])  # 1e13
        plt.ylabel(str(i))
        plt.show()
        df_res.set_index("nr")
        df_res_list.append(df_res)
        '''
        if xy == 0:
            i += -1
            xy = 1
        else:
            xy = 0
        '''
        i += 1

    res = pd.concat(df_res_list)
    del res['ex_nr']

    res = res.drop_duplicates(subset=['nr'], keep='last')
    res.sort_values(by=['nr'], inplace=True, ascending=False)
    res.reset_index(inplace=True)
    return res


#df = pd.read_csv('res_left_deep.txt',sep='|')
#df_rd = pd.read_csv('res_right_deep.txt',sep='|')
df = pd.read_csv('DP/res_left_deep_l2_new.txt',sep='|')
df_rd = pd.read_csv('DP/res_right_deep_l2_new.txt',sep='|')
df_psql_style = pd.read_csv('res_psql_style_left_deep.txt',sep='|')
numofjoins = lambda x: len(x.split('('))
#print()
cost = { 'max':1.e+13, 'min':1.e+6}
sqt = lambda y : ((sqrt(y-cost['min']))/(sqrt(cost['max']-cost['min']))*-10) # SQRT
invers_sqrt  = lambda y : ((y/10*(sqrt(cost['max']-cost['min'])))**2+cost['min'])

#df['numofjoins']=df['ldeepquery'].apply(numofjoins) # -> old
df['numofjoins']=df['resquery'].apply(numofjoins)
df['reward']=df['costs'].apply(sqt)

df_rd['reward']=df_rd['costs'].apply(sqt)
df_psql_style['reward']=df_psql_style['costs'].apply(sqt)
print(df.describe())


#dfkl = df
#dfkl.sort_values(by=['numofjoins'], inplace=True,ascending=False)
#dfkl['query'].to_csv("job_queries_simple_sorted_desc.txt",sep="|",index=False)
print(df.groupby(['numofjoins']).count()['nr'])
#df.sort_values(by=['numofjoins'], inplace=True,ascending=False)


random.seed(7400)
test_is=[]
test_sets = []
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
    test.reset_index(inplace=True)
    test_sets.append(test)
    test_is.append(test_i)

#print(len(test_is))
#print(len(set(test_is)))




path = "results/CrossVal/DQN/"
result_files = ['0_DQN_CM1-postgres-card-job-masking-v0_0_2019-06-20_12-14.txt',
                '1_DQN_CM1-postgres-card-job-masking-v0_0_2019-06-20_13-01.txt',
                '2_DQN_CM1-postgres-card-job-masking-v0_0_2019-06-20_14-29.txt',
                '3_DQN_CM1-postgres-card-job-masking-v0_0_2019-06-20_15-37.txt']
dqn = combineCrossValFiles(path,result_files,test_sets,test_is)

path = "results/CrossVal/DDQN/"
result_files = ['0_DQN_CM1-postgres-card-job-masking-v0_0_2019-06-20_12-34.txt',
                '1_DQN_CM1-postgres-card-job-masking-v0_0_2019-06-20_14-04.txt',
                '2_DQN_CM1-postgres-card-job-masking-v0_0_2019-06-20_15-30.txt',
                '3_DQN_CM1-postgres-card-job-masking-v0_0_2019-06-20_16-33.txt']
ddqn = combineCrossValFiles(path,result_files,test_sets,test_is)

path = "results/CrossVal/PPO/"
result_files = ['0_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-21_07-19.txt',
                '1_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-20_17-42.txt',
                '2_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-20_20-57.txt',
                '3_PPO_CM1-postgres-card-job-masking-v0_0_2019-06-21_00-08.txt']
ppo = combineCrossValFiles(path,result_files,test_sets,test_is)

path = "results/CrossVal/DDQN82/"
result_files = ['DQN_CM1-postgres-card-job-masking-v0_0_2019-06-25_22-49.txt',
#'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-25_22-56.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-25_23-03.txt',
#'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-25_23_11.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-25_23_18.txt',
#'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-25_23_25.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-25_23_33.txt' ]
#'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-25_23_40.txt' ]
ddqn82 = combineCrossValFiles(path,result_files,test_sets,test_is)


path = "results/CrossVal/DDQN11/"
result_files = ['DQN_CM1-postgres-card-job-masking-v0_0_2019-06-25_21-43.txt',
#'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-25_21-47.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-25_21-51.txt',
#'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-25_21-55.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-25_21-59.txt',
#'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-25_22-03.txt',
#'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-25_22-07.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-25_22-11.txt']
ddqn11 = combineCrossValFiles(path,result_files,test_sets,test_is)

df['reward'].plot()
dqn['reward'].plot()
ddqn['reward'].plot()
ppo['reward'].plot()
#df_rd['reward'].plot()
#df_psql_style['reward'].plot()
plt.ylim([-0.55,0])
plt.show()

df_box = pd.DataFrame()
df_box["DP left deep"]=df['reward']
df_box["DP right deep"]=df_rd['reward']
df_box["DQN"]=dqn['reward']
df_box["DDQN"]=ddqn['reward']
df_box["PPO"]=ppo['reward']
#df_box["PSQL style"]=df_psql_style['reward']
df_box.plot.box(showfliers=False)
#plt.ylim([-0.35,0])
plt.show()


df_box = pd.DataFrame()
#df_box["DP left deep"]=df['costs']
#df_box["DP right deep"]=df_rd['costs']
df_box["DQN"]=dqn['costs']
#df_box["DDQN"]=ddqn['costs']
df_box["DDQN82"]=ddqn82['costs']
#df_box["DDQN11"]=ddqn11['costs']
df_box["PPO"]=ppo['costs']
#df_box["PSQL style"]=df_psql_style['reward']
df_box.plot.box(showfliers=False)
#plt.xlabel("?")
plt.ylabel("cost value")
plt.show()



df_box = pd.DataFrame()
#df_box["DP left deep"]=df['costs']
#df_box["DP right deep"]=df_rd['costs']
#df_box["DQN"]=dqn['costs']
#df_box["DDQN"]=ddqn['costs']
df_box["DDQN82"]=ddqn82['costs']
#df_box["DDQN11"]=ddqn11['costs']
df_box["PPO"]=ppo['costs']
#df_box["PSQL style"]=df_psql_style['reward']
df_box.plot.box(showfliers=False)
#plt.xlabel("?")
plt.ylabel("Cost Value")
plt.show()



df_box = pd.DataFrame()
df_box["DDQN82"]=ddqn82['costs']
df_box["PPO"]=ppo['costs']
df_box["DP right deep"]=df_rd['costs']
df_box["DP left deep"]=df['costs']
#df_box["PSQL style"]=df_psql_style['reward']
df_box.plot.box(showfliers=False)
#plt.xlabel("?")
plt.ylabel("Cost Value")
plt.show()


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
df['costs'].plot()
res['costs'].plot()
df_rd['costs'].plot()
plt.show()
'''
