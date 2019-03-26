import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from makeDatasets import MakeDataSets

'''
{u'eegPower': {u'lowGamma': 5144, u'highGamma': 2510, u'highAlpha': 18055, u'delta': 53387,
        u'highBeta': 13139, u'lowAlpha': 27772, u'lowBeta': 6340, u'theta': 81641}, u'poorSignalLevel': 0,
        u'eSense': {u' ': 61, u'attention': 50}}
'''


def removeZerosValue(arr):
    return arr[(arr[['attention', 'meditation']] != 0).all(axis=1)]


def findStdScalar(result):
    scalar = MinMaxScaler()
    scalar.fit(result)
    return scalar.transform(result)


class BuildModel:
    def __init__(self):
        self.file_ = 'model.h5'
        self.columns = ['lowGamma', 'highGamma', 'highAlpha', 'delta', 'highBeta', 'lowAlpha', 'lowBeta',
                        'theta', 'attention', 'meditation']
        if not MakeDataSets().changes:
                self.left = pd.read_csv('./left.csv', names=self.columns)
                self.right = pd.read_csv('./right.csv', names=self.columns)
                self.left['direction'] = 0  # 0 for left
                self.right['direction'] = 1  # 1 for right
        else:
                MakeDataSets().makeDataSet()
                self.left = pd.read_csv('./left.csv', names=self.columns)
                self.right = pd.read_csv('./right.csv', names=self.columns)
                self.left['direction'] = 0  # 0 for left
                self.right['direction'] = 1  # 1 for right

        self.X_train, self.X_test, self.y_train, self.y_test = self.splitDataSet()

    def concatenateFrame(self, frame1, frame2):
        removeZerosValue(self.left)
        removeZerosValue(self.right)
        frame = [frame1, frame2]
        return pd.concat(frame)

    def mergedData(self):
        return self.concatenateFrame(self.left, self.right)

    def splitDataSet(self):
        data = findStdScalar(self.mergedData())
        y = data[:, -1]
        X = data[:, :-1]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def LinearRegression(self):
        reg = LinearRegression().fit(self.X_train, self.y_train)
        y_pred = reg.predict(self.X_test)
        # print(y_pred)
        return [reg, y_pred]

    def printAccuracyOfLinearModel(self):
        out = self.LinearRegression()
        # The coefficients
        print('Coefficients: \n', out[0].coef_)
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(self.y_test, out[1]))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(self.y_test, out[1]))

    def GaussianNaiveBayes(self):
        if not os.path.exists(self.file_):
            print("Model Does not exists...\n Training Begin...\n...Please run again to get specified output..\n")
            gnb = GaussianNB()
            gnb.fit(self.X_train, self.y_train)
            file_d = open(self.file_, 'wb')
            pickle.dump(gnb, file_d)
            del gnb
        else:
            print("Predicting output....\n")
            file_d = open('model.h5', 'rb')
            gnb = pickle.load(file_d)
            y_pred = gnb.predict(self.X_test)
            print(y_pred)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(accuracy)


if __name__ == "__main__":
    bm = BuildModel()
    bm.printAccuracyOfLinearModel()
    bm.GaussianNaiveBayes()

"""
def rawToVoltage(val):
    for i in range(len(val)):
        dx = 1 / ((int(val[i]) * (1.8 / 4096)) / 2000)
        print(dx)
"""
