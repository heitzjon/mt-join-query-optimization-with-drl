from math import log, sqrt
import matplotlib.pyplot as plt
import numpy as np

################################################################
# Reward plot

cost = {'max': 1.e+13, 'min': 1.e+6}
y_array = np.arange(cost['min']+2,1.e18,1.e+12)
sqt = lambda y2 : map(lambda y : ((sqrt(y-cost['min']))/(sqrt(cost['max']-cost['min']))*-10),y2) # SQRT

sq = list(sqt(y_array))

for i in range(0,len(sq)):
    if sq[i]<-10:sq[i]=-10


plt.plot(sq,y_array)

plt.ylim(-0.5*1e11,2*1e13)
#plt.xlim(-10,0)
#plt.xscale('log')
plt.ylabel("Cost Value")
plt.xlabel("Reward")

plt.show()