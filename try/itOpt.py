import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

df = pd.read_excel("../test/ev.xlsx", index_col= 0)
ei = df[["ev"]].to_numpy()
print(ei)
print(ei.shape)


beta = []
#partition function
t=[]
for k in range(10,40,2):
        for i in range(70):
            ti = math.exp(-1*(float(k)/10)*ei[i])
            if ti<(10**(-10)):
                t.append(i)
                beta.append((float(k)/10))
                break
print(t)
print(beta)
plt.plot(beta, t, '-o')
plt.xticks(np.arange(1,4,0.2))
plt.show()
