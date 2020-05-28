import math
import numpy as np
import pandas as pd
'''
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
'''

#importing data
df1 = pd.read_csv("occ_L8_N4.dat", sep='\s+', names=['it','energy',1,2,3,4,5,6,7,8])
df2 = pd.read_csv("spectrum_L8_N4.dat", sep='\s+', names=['en', 'it '])

occ = df1[[1, 2, 3, 4, 5, 6, 7, 8]].to_numpy()
en = df2[['en']].to_numpy()
uen = df1[['energy']].to_numpy()

occAvg = np.zeros(8000).reshape(1000,8)
uenAvg = np.zeros(1000)
z = np.zeros(1000)
temp = 0
beta = 2

for i in range(0,70000):
    occ[i]=occ[i]*(math.exp(-1*beta*en[i]))
    uen[i]=uen[i]*(math.exp(-1*beta*en[i]))
    occAvg[i/70] = occAvg[i/70] + occ[i]  
    uenAvg[i/70] = uenAvg[i/70] + uen[i] 
    temp = temp + math.exp(-1*beta*en[i])
    if (i+1)%70 == 0:
        z[((i+1)/70)-1] = temp
        occAvg[i/70] = occAvg[i/70]*float(1/temp)
        uenAvg[i/70] = uenAvg[i/70]*float(1/temp)
        temp = 0
    
print(occAvg)
print(uenAvg)



'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(occAvg, uenAvg, test_size=0.3)
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))
	model.add(Dense(128, kernel_initializer='normal',activation='linear'))
	model.add(Dense(128, kernel_initializer='normal',activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation = 'linear'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10)
#results = cross_val_score(pipeline, occ_arr, en_err, cv=kfold)
#print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

pipeline.fit(X_train, y_train)
y_pred= pipeline.predict(X_test)
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
'''
