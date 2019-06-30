import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import random
from statistics import mean



df = pd.read_csv('res_left_deep.txt',sep='|')
df_leftdeep = pd.read_csv('res_left_deep.txt',sep='|')
df_greedy = pd.read_csv('res_greedy_left_deep.txt',sep='|')
#df_greedy = pd.read_csv('res_psql_style_left_deep6.txt',sep='|')
df_psql_style = pd.read_csv('res_psql_style_left_deep.txt',sep='|')
df_postgres = pd.read_csv('res_postgres_stat.txt',sep='|')
#df = df_greedy
numofjoins = lambda x: len(x.split('('))-1
#print()
cost = { 'max':1.e+13, 'min':1.e+6}
sqt = lambda y : ((sqrt(y-cost['min']))/(sqrt(cost['max']-cost['min']))*-10) # SQRT
invers_sqrt  = lambda y : ((y/10*(sqrt(cost['max']-cost['min'])))**2+cost['min'])

df['numofjoins']=df['ldeepquery'].apply(numofjoins)
df['reward']=df['costs'].apply(sqt)
print(df.describe())

i_rejoin = [0,1,2,3,28,42,46,51,56,80]
df_rejoin = df_leftdeep.iloc[i_rejoin]
df_rejoin_greedy = df_greedy.iloc[i_rejoin]

#df_rejoin["costs"].plot()
#df_rejoin_greedy["costs"].plot()
#plt.show()

i_rejoin_train = []
for x in range(0,len(df)):
    if x not in i_rejoin:
        i_rejoin_train.append(x)
df_rejoin_train = df.iloc[i_rejoin_train]
df_rejoin_test = df_rejoin

#df_rejoin_test["query"].to_csv("job_queries_simple_rejoin_test.txt",header=False, sep="|",index=False)
#df_rejoin_train["query"].to_csv("job_queries_simple_rejoin_train.txt",header=False, sep="|",index=False)


#dfkl = df
#dfkl.sort_values(by=['numofjoins'], inplace=True,ascending=False)
#dfkl['query'].to_csv("job_queries_simple_sorted_desc.txt",sep="|",index=False)
print(df.groupby(['numofjoins']).count()['nr'])
#df.sort_values(by=['numofjoins'], inplace=True,ascending=False)


#del df_rejoin['index']
df_rejoin.reset_index(inplace=True)
#del df_rejoin_greedy['index']
df_rejoin_greedy.reset_index(inplace=True)

result_files = [
    'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-18_15-58.txt'
    #,'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-18_16-19.txt' # okish
    #,'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-18_17-05.txt' # okish
    #,'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-18_17-36.txt' #too bad
    #,'PPO_CM1-postgres-card-job-masking-v0_0_2019-06-18_18-07.txt' #too bad
]
path = 'results/ReJoin/'
for filename in result_files:
    df_rejoin["costs"].plot(label="DP")
    df_rejoin_greedy["costs"].plot(label="greedy left deep")
    file = open(path+filename,'r')
    raw_rew=file.read().replace('\nEpisode reward ',',').replace('\n(',',(')
    res = []
    for l in raw_rew.split('\n'):
        x = l.split(",")
        res.append([int(x[0]),i_rejoin[int(x[0])],x[1],float(x[2])])

    df_res = pd.DataFrame(res,columns=["id","nr","plan","reward"])
    df_res["costs"] = df_res['reward'].apply(invers_sqrt)
    print(df_res.describe())

    df_res["costs"].plot(label="FODB PPO")
    #df_res["reward"].plot()
    #df_res.plot.scatter(x='nr', y='reward', c='r')
    #plt.scatter(df_res['nr'], df_res['reward'], c='r')
    print(df_res.where(df_res['reward']<=-0.30).dropna())
    #plt.ylim([-0.55,0]) #1e13
    plt.ylim([0,0.1e11]) #1e13
    plt.legend(loc='upper left')
    plt.show()
    #x = df_res["costs"]/df_rejoin_greedy["costs"].mean()*100
    x = df_res["costs"] / df_rejoin["costs"].mean() * 100
    #x = df_res["costs"] / df_leftdeep["costs"].mean() * 100
    x.plot.box(label="FOOP")
    x =[0,1,2]
    y = [100,100,100]
    plt.plot(x,y,label="left deep ")
    y = [80,80,80]
    plt.plot(x,y,label="ReJoin")

    plt.ylim([0, 200])
    #plt.xlabel("?")
    plt.ylabel("%")
    plt.legend(loc='lower left')
    plt.show()
    print(df_rejoin.describe())
    print(df_rejoin_greedy.describe())

# normalize
norm = lambda y : (y-y.min())/(y.max()-y.min())

df_leftdeep_old = pd.read_csv('res_left_deep.txt',sep='|')

df_postgres = pd.read_csv('res_postgres_stat.txt',sep='|')

#df_leftdeep['numofjoins']=df_leftdeep['resquery'].apply(numofjoins)
df_leftdeep_old['numofjoins']=df_leftdeep_old['ldeepquery'].apply(numofjoins)



coef = df_leftdeep_old['costs'].max()-df_leftdeep_old['costs'].min().copy()
min = df_leftdeep_old['costs'].min().copy()

print(df_postgres.describe())
norm_df = pd.DataFrame()
#norm_df['postgres_cost'] = norm(df_postgres['est_cost'])
i_rejoin = [0,1,2,3,28,42,46,51,56,80]
#df_postgres_simple = df_postgres_simple.iloc[i_rejoin]
#df_leftdeep = df_leftdeep.iloc[i_rejoin]
#norm_df['postgres_cost_mean'] = norm(df_postgres_simple['est_cost_mean'])
#norm_df['PostgreSQL']=norm_df[['postgres_cost_simple', 'postgres_cost_s_large']].mean(axis=1)

#norm_df['greedy_cost'] = (df_greedy['costs']-min)/coef#norm(df_greedy['costs'])

#norm_df['postgres_style_cost'] = norm(df_psql_style['costs'])
norm_df['DP left-deep'] = norm(df_leftdeep_old['costs'])

norm_df['PostgreSQL']=norm_df['postgres_cost_simple']
del norm_df['postgres_cost_simple']
del norm_df['postgres_cost_s_large']
#norm_df['Left-Deep DP l1'] = norm(df_leftdeep['costs'])
#norm_df['Left-Deep DP l2'] = norm(df_leftdeep_2['costs'])
#norm_df['Right-Deep DP l1'] = norm(df_rightdeep['costs'])
#norm_df['Right-Deep DP l2'] = norm(df_rightdeep_2['costs'])
#norm_df['postgres_style_cost'] = df_psql_style['costs']
#norm_df['dp_ld_cost'] = df_leftdeep['costs']
norm_df['numofjoins'] = df_leftdeep_old['numofjoins']
#x = (df_res['costs']-df_leftdeep['costs'].min())/(df_leftdeep['costs'].max()-df_leftdeep['costs'].min())
#x = (df_res['costs']-min)/coef
#norm_df = norm_df.iloc[i_rejoin]
norm_df.reset_index(inplace=True)

#norm_df["FOOP"] = x
#norm_df["FOOP"] = norm(df_res['costs'])
norm_df.sort_values(by=['numofjoins','DP left-deep'], inplace=True,ascending=True)
#norm_df.sort_values(by=['numofjoins'], inplace=True,ascending=True)
print(df_res['costs']-df_leftdeep['costs'].min())
print(df_leftdeep['costs'].max()-df_leftdeep['costs'].min())
print(norm_df)
norm_df.reset_index(inplace=True)
del norm_df['numofjoins']
del norm_df['index']
del norm_df['level_0']
norm_df.plot()
plt.legend(loc='upper left')
plt.xlabel("# Of Joins Per Query")
plt.show()



ppo = mean([9.426, 9.120, 9.107,9.974 ,9.484])/900*1000
ddqn = mean([19.024, 18.763,18.905,18.806, 18.826])/900*1000
dqn = mean([6.670,6.781,6.806,6.658,6.646])/900*1000

ppo_el = ppo*5
ddqn_el = ddqn*5
dqn_el = dqn*5

df_leftdeep_old = pd.read_csv('res_left_deep.txt',sep='|')
df_leftdeep_old['numofjoins']=df_leftdeep_old['ldeepquery'].apply(numofjoins)

bar_df = pd.DataFrame()
bar_df['DP left-deep'] = df_leftdeep_old['planningtime']*1000
bar_df['numofjoins'] = df_leftdeep_old['numofjoins']
#bar_df['DQN'] = bar_df['numofjoins']*dqn
#bar_df['DDQN'] = bar_df['numofjoins']*ddqn
bar_df['FOOP PPO'] = bar_df['numofjoins']*ppo
#bar_df['DQN with EL'] = bar_df['numofjoins']*dqn_el
#bar_df['DDQN with EL'] = bar_df['numofjoins']*ddqn_el
bar_df['FOOP PPO with EL'] = bar_df['numofjoins']*ppo_el




bar_df.groupby('numofjoins').max().plot.barh()
plt.legend(loc='lower right')
plt.semilogx()
#plt.xlim([2000, 20000])

plt.xlabel("Optimization Time [ms] ")
plt.ylabel("# Of Joins Per Query ")
#plt.semilogx()
plt.show()


'''
Optimization Latency PPO

real	0m9.426s
user	0m9.746s
sys	0m1.731s

real	0m9.120s
user	0m9.729s
sys	0m1.729s

real	0m9.107s
user	0m9.723s
sys	0m1.555s

real	0m9.974s
user	0m9.723s
sys	0m1.664s

real	0m9.484s
user	0m9.923s
sys	0m1.725s

DDQN
real	0m19.024s
user	0m17.811s
sys	0m4.052s

real	0m18.763s
user	0m17.550s
sys	0m4.235s

real	0m18.905s
user	0m17.675s
sys	0m4.089s

real	0m18.806s
user	0m17.719s
sys	0m4.060s

real	0m18.826s
user	0m17.802s
sys	0m3.955s


DQN
real	0m6.670s
user	0m6.961s
sys	0m1.491s

real	0m6.781s
user	0m7.194s
sys	0m1.491s

real	0m6.806s
user	0m7.116s
sys	0m1.504s

real	0m6.658s
user	0m7.190s
sys	0m1.594s

real	0m6.646s
user	0m7.006s
sys	0m1.776s


'''