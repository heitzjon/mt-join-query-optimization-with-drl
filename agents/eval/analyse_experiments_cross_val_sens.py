import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import random
from functools import reduce


def combineCrossValFiles(path, result_files,test_sets,test_is):
    i = 0
    xy = 0
    df_res_list = []
    for filename in result_files:

        file = open(path + filename, 'r')
        raw_rew = file.read().replace('\nEpisode reward ', ',')
        res = []
        for l in raw_rew.split('\n'):
            x = l.split(",")
            if x[0] is not '': res.append([int(x[0]), float(x[1])])

        df_res = pd.DataFrame(res, columns=["ex_nr", "reward"])
        df_res["costs"] = df_res['reward'].apply(invers_sqrt)
        #print("____________________________________________")
        #print("split: "+str(i)+" num: "+str(xy))
        df_res["nr"] = test_is[i]
        # print(df_res.describe())


        #print(df_res.where(df_res['reward'] <= -0.35).dropna())

        '''
        test_sets[i]["reward"].plot()
        # df_res["costs"].plot()
        df_res["reward"].plot()
        # df_res.plot.scatter(x='nr', y='reward', c='r')
        # plt.scatter(df_res['nr'], df_res['reward'], c='r')
        plt.ylim([-0.55, 0])  # 1e13
        plt.ylabel(str(i))
        plt.show()
        print(df_res['reward'].describe())
        '''
        df_res.set_index("nr")
        df_res_list.append(df_res)
        '''
        if xy < 5:
            #i += -1
            xy += 1
        else:
            xy = 0

            i += 1
        '''
        i += 1

    res = pd.concat(df_res_list)
    del res['ex_nr']

    res = res.drop_duplicates(subset=['nr'], keep='last')
    res.sort_values(by=['nr'], inplace=True, ascending=False)
    res.reset_index(inplace=True)
    return res


def combineCrossValFilesBest(path, result_files, test_sets, test_is,num):
    i = 0
    xy = 0
    df_res_list = []
    res = []
    j=0
    for filename in result_files:
        #test_sets[i]["reward"].plot()
        file = open(path + filename, 'r')
        raw_rew = file.read().replace('\nEpisode reward ', ',')
        for l in raw_rew.split('\n'):
            x = l.split(",")
            if x[0] is not '': res.append([int(x[0]), float(x[1])])
        if j < num-1:
            j+=1
        else:
            j=0
            #print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
            #print(res)
            df_res = pd.DataFrame(res, columns=["ex_nr", "reward"])
            res = []
            df_res["costs"] = df_res['reward'].apply(invers_sqrt)
            df_res = df_res.groupby("ex_nr").agg('min')
            print(df_res.describe())
            #print(df_res.describe())
            #print(i)
            df_res["nr"] = test_is[i]
            # print(df_res.describe())

            # df_res["costs"].plot()
            #df_res["reward"].plot()
            # df_res.plot.scatter(x='nr', y='reward', c='r')
            # plt.scatter(df_res['nr'], df_res['reward'], c='r')
            #print(df_res.where(df_res['reward'] <= -0.35).dropna())
            #plt.ylim([-0.55, 0])  # 1e13
            #plt.ylabel(str(i))
            #plt.show()
            df_res.set_index("nr")
            print(df_res.describe())
            df_res_list.append(df_res)
            i+=1
        '''
        if xy < 4:
            #i += -1
            xy += 1
        else:
            xy = 0

            i += 1
        '''

    res = pd.concat(df_res_list)
    #del res['ex_nr']

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


# normalize
norm = lambda y : (y-y.min())/(y.max()-y.min())


df = pd.read_csv('res_left_deep.txt',sep='|')
df_postgres_simple = pd.read_csv('res_simple_postgres_stat.txt',sep='|')


#SENSITIVITY ANALYSIS RESULTS
numofjoins = lambda x: len(x.split('('))-1
df['numofjoins'] = df['ldeepquery'].apply(numofjoins)
df['reward']=df['costs'].apply(sqt)
df_s = df.loc[(df['numofjoins'] <= 6)]
df_m = df.loc[(df['numofjoins'] >= 7) & (df['numofjoins'] <= 10)]
df_l = df.loc[(df['numofjoins'] >= 11)]


df['postgres_cost_s'] = norm(df_postgres_simple['est_cost'])*(df['costs'].max()-df['costs'].min())+df['costs'].min()
df['postgres_cost_s_large'] = norm(df_postgres_simple['est_cost_large'])*(df['costs'].max()-df['costs'].min())+df['costs'].min()

#random.seed(7400)

random.seed(8400)


rand_ord_s = list(range(0, len(df_s)))+random.sample(range(0, len(df_s)), 8)
rand_ord_m = list(range(0, len(df_m)))+random.sample(range(0, len(df_m)), 8)
rand_ord_l = list(range(0, len(df_l)))+random.sample(range(0, len(df_l)), 4)

random.shuffle(rand_ord_s)
random.shuffle(rand_ord_m)

pointer_s = 0
pointer_m = 0
pointer_l = 0


test_is_sens=[]
test_sets_sens = []

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
    #df_test = df.loc[df['nr'].isin(test_nr)]
    df_test = df.loc[df['nr'].reindex(test_nr)]

    #df_train = df.loc[~df['nr'].isin(test_nr)]
    df_test.reset_index(inplace=True)
    test_sets_sens.append(df_test)
    test_is_sens.append(test_nr)


path = "results/Sens/DQN/"
result_files = [#'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_17-38.txt',
'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_17-51.txt',
#'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_18-03.txt',
#'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_18-12.txt',
#'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_18-23.txt',
#'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_18-34.txt',
#'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_18-46.txt',
#'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_18-57.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_19-09.txt',
#'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_19-21.txt',
#'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-26_19-33.txt',
#'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-26_19-45.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-26_19-56.txt',
#'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-26_20-08.txt',
#'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-26_20-20.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-26_20-32.txt']#,
#'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-26_20-43.txt',
#'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-26_20-52.txt',
#'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-26_21-01.txt',
#'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-26_21-10.txt' ]
dqn_s = combineCrossValFiles(path,result_files,test_sets_sens,test_is_sens)



path = "results/Sens/DDQN/"
result_files = ['DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_22-43.txt',
#'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_22-50.txt',
#'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_22-57.txt',
#'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_23-04.txt',
#'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_23-12.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_23-19.txt',
#'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_23-26.txt',
#'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_23-33.txt',
#'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_23-41.txt',
#'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_23-48.txt',
#'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-26_23-55.txt',
#'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-27_00-03.txt',
#'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-27_00-10.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-27_00-17.txt',
#'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-27_00-24.txt',
#'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-27_00-32.txt',
#'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-27_00-39.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-27_00-46.txt']#,
#'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-27_00-53.txt',
#'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-27_01-01.txt' ]
ddqn_s = combineCrossValFiles(path,result_files,test_sets_sens,test_is_sens)



path = "results/Sens/PPO/"
result_files = [#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_01-08.txt',
#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_01-32.txt',
#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_01-57.txt',
'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_02-20.txt',
#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_02-45.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_03-09.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_03-32.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_03-56.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_04-19.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_04-43.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-27_05-08.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-27_05-31.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-27_05-56.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-27_06-20.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-27_06-45.txt',
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-27_07-09.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-27_07-34.txt']#,
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-27_07-59.txt',
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-27_08-28.txt',
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-27_08-58.txt' ]
ppo_s = combineCrossValFiles(path,result_files,test_sets_sens,test_is_sens)


path = "results/Sens/DQN/"
result_files = ['DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_17-38.txt',
'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_17-51.txt',
'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_18-03.txt',
'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_18-12.txt',
'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_18-23.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_18-34.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_18-46.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_18-57.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_19-09.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_19-21.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-26_19-33.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-26_19-45.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-26_19-56.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-26_20-08.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-26_20-20.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-26_20-32.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-26_20-43.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-26_20-52.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-26_21-01.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-26_21-10.txt' ]
dqn_e = combineCrossValFilesBest(path,result_files,test_sets_sens,test_is_sens,5)


path = "results/Sens/DDQN/"
result_files = ['DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_22-43.txt',
'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_22-50.txt',
'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_22-57.txt',
'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_23-04.txt',
'DQN_CM1-postgres-card-job-masking-v0_0_2019-06-26_23-12.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_23-19.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_23-26.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_23-33.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_23-41.txt',
'DQN_CM1-postgres-card-job-masking-v1_0_2019-06-26_23-48.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-26_23-55.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-27_00-03.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-27_00-10.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-27_00-17.txt',
'DQN_CM1-postgres-card-job-masking-v2_0_2019-06-27_00-24.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-27_00-32.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-27_00-39.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-27_00-46.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-27_00-53.txt',
'DQN_CM1-postgres-card-job-masking-v3_0_2019-06-27_01-01.txt' ]
ddqn_e = combineCrossValFilesBest(path,result_files,test_sets_sens,test_is_sens,5)



path = "results/Sens/PPO/"
result_files = ['PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_01-08.txt',
'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_01-32.txt',
'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_01-57.txt',
'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_02-20.txt',
'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_02-45.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_03-09.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_03-32.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_03-56.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_04-19.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_04-43.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-27_05-08.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-27_05-31.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-27_05-56.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-27_06-20.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-27_06-45.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-27_07-09.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-27_07-34.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-27_07-59.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-27_08-28.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-27_08-58.txt' ]
ppo_e = combineCrossValFilesBest(path,result_files,test_sets_sens,test_is_sens,5)



path = "results/Sens/PPO_2/"
result_files = [#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_18-54.txt',
#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_19-14.txt',
#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_19-41.txt',
#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_20-10.txt',
#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_20-38.txt',
'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_21-08.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_21-38.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_22-05.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_22-32.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_22-58.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_23-22.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_23-48.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_00-13.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_00-37.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_01-01.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_01-25.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_01-49.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_02-13.txt',
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_02-37.txt',
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_03-01.txt',
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_03-24.txt',
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_03-48.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_04-12.txt']#,
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_04-34.txt' ]
ppo_s2 = combineCrossValFiles(path,result_files,test_sets_sens,test_is_sens)#,6)


path = "results/Sens/PPO_2/"
result_files = [#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_18-54.txt',
#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_19-14.txt',
'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_19-41.txt',
#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_20-10.txt',
#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_20-38.txt',
#'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_21-08.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_21-38.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_22-05.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_22-32.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_22-58.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_23-22.txt',
#'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_23-48.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_00-13.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_00-37.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_01-01.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_01-25.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_01-49.txt',
#'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_02-13.txt',
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_02-37.txt',
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_03-01.txt',
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_03-24.txt',
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_03-48.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_04-12.txt']#,
#'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_04-34.txt' ]
ppo_s3 = combineCrossValFiles(path,result_files,test_sets_sens,test_is_sens)#,6)


path = "results/Sens/PPO_2/"
result_files = ['PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_18-54.txt',
'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_19-14.txt',
'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_19-41.txt',
'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_20-10.txt',
'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-27_21-08.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_22-05.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_22-32.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_22-58.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_23-22.txt',
'PPO_CM1-postgres-card-job-masking-v1_0_2019-06-27_23-48.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_00-13.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_00-37.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_01-01.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_01-25.txt',
'PPO_CM1-postgres-card-job-masking-v2_0_2019-06-28_01-49.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_02-37.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_03-01.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_03-24.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_03-48.txt',
'PPO_CM1-postgres-card-job-masking-v3_0_2019-06-28_04-12.txt']
ppo_e2 = combineCrossValFilesBest(path,result_files,test_sets_sens,test_is_sens,5)


print("SDFH")
#print(dqn_e.head())
print("left deep")
print(df['costs'].describe())
print("dqn")
print(dqn_e['costs'].describe())
print("ppo")
print(ppo_e['costs'].describe())
print(ppo['costs'].describe())
print(dqn['costs'].describe())

'''
df['reward'].plot()
dqn['reward'].plot()
ddqn['reward'].plot()
ppo['reward'].plot()
#df_rd['reward'].plot()
#df_psql_style['reward'].plot()
plt.ylim([-0.55,0])
plt.show()
'''

df['costs'].plot(label="dp")
dqn_e['costs'].plot(label="dqn")
ddqn_e['costs'].plot(label="ddqn")
ppo_e['costs'].plot(label="ppo")
plt.show()


'''
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
'''

'''
fig = plt.figure()

# Divide the figure into a 1x2 grid, and give me the first section
ax1 = fig.add_subplot(131)
# Divide the figure into a 1x2 grid, and give me the second section
ax2 = fig.add_subplot(132)
# Divide the figure into a 1x2 grid, and give me the second section
ax3 = fig.add_subplot(133)
'''
df_box = pd.DataFrame()
#df_box["PostgreSQL*"]=df['postgres_cost_s']
#df_box["DP right deep"]=df_rd['costs']
#df_box["DQN"]=dqn['costs']
#df_box["DQN_s"]=dqn_s['costs']
df_box["DQN"]=dqn['costs']
#df_box["DDQN82"]=ddqn82['costs']
df_box["DDQN"]=ddqn82['costs']
#df_box["DDQN11"]=ddqn11['costs']
#df_box["PPO"]=ppo['costs']
#df_box["PSQL style"]=df_psql_style['reward']
df_box.plot.box(showfliers=False)
#plt.xlabel("?")
plt.grid(axis='y')
plt.ylabel("Cost Value")
plt.show()



df_box1 = pd.DataFrame()
df_box2 = pd.DataFrame()
df_box3 = pd.DataFrame()
df_box1["DQN"]=dqn['costs']
df_box2["DQN"]=dqn_s['costs']
df_box3["DQN"]=dqn_e['costs']
df_box1["DDQN"]=ddqn82['costs']
df_box2["DDQN"]=ddqn_s['costs']
df_box3["DDQN"]=ddqn_e['costs']
#df_box["DDQN11"]=ddqn11['costs']
df_box1["PPO"]=ppo['costs']
df_box2["PPO"]=ppo_s['costs']
#df_box["PPO_s2"]=ppo_s2['costs']
#df_box["PPO_s3"]=ppo_s3['costs']
df_box3["PPO"]=ppo_e['costs']
#df_box["PPO_e2"]=ppo_e['costs']
#df_box["PSQL style"]=df_psql_style['reward']
#df_box.plot.box(showfliers=False)
#plt.xlabel("?")
#plt.ylabel("cost value")
#plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(16,6))
#plt.subplot(131)
df_box1.plot.box(showfliers=False, ax=ax1)
ax1.set_xlabel('On Random Training Data')
ax1.set_ylabel("Cost Value")
ax1.grid(axis='y')


df_box2.plot.box(showfliers=False, ax=ax2)
ax2.set_xlabel('After Sensitivity Analysis')
ax2.grid(axis='y')

df_box3.plot.box(showfliers=False, ax=ax3)
ax3.set_xlabel('With Ensemble Learning')
ax3.grid(axis='y')
plt.show()

df_box = pd.DataFrame()
#df_box["PostgreSQL*"]=df['postgres_cost_s']
#df_box["DP right deep"]=df_rd['costs']
#df_box["DQN"]=dqn['costs']
#df_box["DQN_s"]=dqn_s['costs']
df_box["DQN"]=dqn_e['costs']
#df_box["DDQN82"]=ddqn82['costs']
df_box["DDQN"]=ddqn_e['costs']
df_box["PPO"]=ppo_e['costs']
df_box["DP left deep"]=df['costs']
#df_box["DDQN11"]=ddqn11['costs']
#df_box["PPO"]=ppo['costs']
#df_box["PSQL style"]=df_psql_style['reward']
df_box.plot.box(showfliers=False)
#plt.xlabel("?")
plt.grid(axis='y')
plt.ylabel("Cost Value")
plt.show()

df_graph = df.copy()
df_graph['ppo_cost'] = ppo_e['costs']
df_graph['ddqn_cost'] = ddqn_e['costs']
df_graph['dqn_cost'] = dqn_e['costs']
df_graph.sort_values(by=['numofjoins','costs'], inplace=True,ascending=True)
df_graph.reset_index(inplace=True)
df_graph.set_index('numofjoins')

df_graph['costs'].plot(label="DP left-deep")
df_graph['ppo_cost'].plot(label="FOOP PPO")

#df_graph['ddqn_cost'].plot(label="dqn")
#df_graph['dqn_cost'].plot(label="ddqn")
plt.ylim(0,2.e10)
plt.ylabel("Cost Value")
plt.xlabel("# Of Joins Per Query")
plt.legend(loc='upper left')
plt.show()

print(df.head())
