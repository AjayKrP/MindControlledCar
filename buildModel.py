import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

'''
{u'eegPower': {u'lowGamma': 5144, u'highGamma': 2510, u'highAlpha': 18055, u'delta': 53387,
        u'highBeta': 13139, u'lowAlpha': 27772, u'lowBeta': 6340, u'theta': 81641}, u'poorSignalLevel': 0,
        u'eSense': {u' ': 61, u'attention': 50}}
'''

columns = ['lowGamma', 'highGamma', 'highAlpha', 'delta', 'highBeta', 'lowAlpha', 'lowBeta',  'theta', 'attention' ,'meditation']
left = pd.read_csv('./left.csv', names=columns)
right = pd.read_csv('./right.csv', names=columns)
left['direction'] = 0 # 0 for left
right['direction'] = 1 # 1 for right

left = left[(left[['attention','meditation']] != 0).all(axis=1)]
right = right[(right[['attention','meditation']] != 0).all(axis=1)]

frame = [left, right]
result = pd.concat(frame)

scaler = MinMaxScaler()
scaler.fit(result)
data = scaler.transform(result)

y = data[:, -1]
X = data[:, :-1]
#print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# reg = LinearRegression().fit(X_train, y_train)
# y_pred = reg.predict(X_test)
# #print(y_pred)

# # The coefficients
# print('Coefficients: \n', reg.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(y_test, y_pred))


def buildModel(file_):
        if not os.path.exists(file_):
                gnb = GaussianNB()
                gnb.fit(X_train, y_train)
                file_d = open(file_, 'wb')
                pickle.dump(gnb, file_d)
                del gnb

def predictDirection(arg_):
        file_d = open('model.h5', 'rb')
        gnb = pickle.load(file_d)
        y_pred = gnb.predict(X_test)
        #print(y_pred)
        print(gnb.predict([arg_]))
        if (gnb.predict([arg_]) == 0):
                return 'left'
        else:
                return 'right'
        #accuracy = accuracy_score(y_test, y_pred)
        #print(accuracy)


def rawToVoltage(val):
    for i in range(len(val)):
        dx = 1/((int(val[i])*(1.8/4096))/2000)
        print(dx)


