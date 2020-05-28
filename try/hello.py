import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#importing the excel 
df = pd.read_excel("../test/new_occ.xlsx", index_col= 0)


#shuffling
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())
print(df.shape)

occ_arr = df[[1, 2, 3, 4, 5, 6, 7, 8]].to_numpy()
en_err = df[["energy"]].to_numpy()
print(occ_arr)
print(en_err)

'''
from sklearn.preprocessing import  MinMaxScaler
sc= MinMaxScaler()
occ_arr = sc.fit_transform(occ_arr)
en_err = en_err.reshape(-1,1)
en_err =sc.fit_transform(en_err)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(occ_arr,en_err, test_size=0.3)

from keras import Sequential
from keras.layers import Dense
def build_regressor():
 	regressor = Sequential()
	regressor.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))
	regressor.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	regressor.compile(loss='mean_squared_error', optimizer='adam')
	return regressor

from keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=build_regressor, batch_size=32,epochs=100)

results=regressor.fit(X_train,y_train)

y_pred= regressor.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(occ_arr, en_err, test_size=0.3)
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))
	model.add(Dense(256, kernel_initializer='normal',activation='relu'))
	model.add(Dense(256, kernel_initializer='normal',activation='relu'))
	model.add(Dense(256, kernel_initializer='normal',activation='relu'))
	model.add(Dense(256, kernel_initializer='normal',activation='relu'))
	model.add(Dense(256, kernel_initializer='normal',activation='relu'))
	#model.add(Dense(256, kernel_initializer='normal',activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation = 'linear'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
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
