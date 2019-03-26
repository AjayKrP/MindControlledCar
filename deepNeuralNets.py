from __future__ import absolute_import, division, print_function

# Helper libraries
import numpy as np
import pandas as pd
# TensorFlow and tf.keras
import tensorflow as tf
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from buildModel import BuildModel

print(tf.__version__)
tf.enable_eager_execution()


class DeepNeuralNetwork:
    def findDerivate(self, arr):
        temp = []
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            if arr[i] < arr[i + 1]:
                temp.append(1)
            elif arr[i] == arr[i + 1]:
                temp.append(0)
            elif arr[i] > arr[i + 1]:
                temp.append(-1)

        temp.append(1)
        return temp

    def findEnvelopes(self, s):
        q_u = np.zeros(s.shape)
        q_l = np.zeros(s.shape)

        # Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting
        # point for both the upper and lower envelope models.

        u_x = [0, ]
        u_y = [s[0], ]

        l_x = [0, ]
        l_y = [s[0], ]

        # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

        for k in range(1, len(s) - 1):
            if (np.sign(s[k] - s[k - 1]) == 1) and (np.sign(s[k] - s[k + 1]) == 1):
                u_x.append(k)
                u_y.append(s[k])

            if (np.sign(s[k] - s[k - 1]) == -1) and ((np.sign(s[k] - s[k + 1])) == -1):
                l_x.append(k)
                l_y.append(s[k])

        # Append the last value of (s) to the interpolating values. This forces the model to use the same ending point
        # for both the upper and lower envelope models.

        u_x.append(len(s) - 1)
        u_y.append(s[-1])

        l_x.append(len(s) - 1)
        l_y.append(s[-1])

        # Fit suitable models to the data. Here I am using cubic splines, similarly to the MATLAB example given in the
        # question.

        u_p = interp1d(u_x, u_y, kind='cubic', bounds_error=False, fill_value=0.0)
        l_p = interp1d(l_x, l_y, kind='cubic', bounds_error=False, fill_value=0.0)

        # Evaluate each model over the domain of (s)
        temp = []
        for k in range(0, len(s)):
            q_u[k] = u_p(k)
            q_l[k] = l_p(k)
            temp.append((q_u[k] + q_l[k]) / 2)

        temp.append(temp[len(temp) - 1])
        return np.array(temp)

    def startBuildingModel(self, attention, highBeta, lowBeta):
        attentionDerivate = self.findDerivate(attention)
        beta = []
        highBeta = np.array(highBeta)
        lowBeta = np.array(lowBeta)
        for i in range(len(highBeta)):
            beta.append((highBeta[i] + lowBeta[i]) / 2)

        beta = np.array(beta)
        betaDerivate = self.findDerivate(beta)
        betaEnvelope = self.findEnvelopes(beta)

        print(attention.shape)
        attentionDerivate = pd.DataFrame(attentionDerivate, columns=['attentionD'])
        print(attentionDerivate)
        print(attentionDerivate.shape)
        print(beta.shape)
        frame = [pd.DataFrame(attention, columns=['attention']), attentionDerivate, \
                 pd.DataFrame(beta, columns=['beta']), pd.DataFrame(betaDerivate, columns=['betaD']), \
                 pd.DataFrame(betaEnvelope, columns=['betaE']), pd.DataFrame(lowBeta, columns=['lowBeta']), \
                 pd.DataFrame(highBeta, columns=['highBeta'])]

        result = pd.concat(frame)
        # print(result)
        scaler = MinMaxScaler()
        scaler.fit(result)
        data = scaler.transform(result)
        data = pd.DataFrame(data, columns=result.columns, index=result.index);
        data.dropna(axis=0);

        # print(data)
        print(data)
        y = data[:, 0]
        X = data[:, 1:]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print(X_train.shape)

        tf.convert_to_tensor(X_train, dtype=tf.float32)
        tf.convert_to_tensor(X_test, dtype=tf.float32)
        tf.convert_to_tensor(y_train, dtype=tf.float32)
        tf.convert_to_tensor(y_test, dtype=tf.float32)

        print(X_train.shape)
        model = keras.Sequential([
            keras.layers.Dense(units=7, input_shape=(6,)),
            keras.layers.Dense(units=5, activation=tf.nn.relu),
            keras.layers.Dense(units=3, activation=tf.nn.relu),
            keras.layers.Dense(units=2, activation=tf.nn.relu)
        ])

        X_train.astype(float)
        y_train.astype(float)
        X_test.astype(float)
        y_test.astype(float)

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        print("Shape: ", X_train.shape)
        print("Shape: ", y_train.shape)

        model.fit(X_train, y_train, epochs=50)
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print('Test accuracy:', test_acc)

        test = [[[5697, 958, 10192, 213943, 3255, 1942, 12090, 42660, 80, 75]]]
        predictions = model.predict(test)
        print(predictions)


if __name__ == "__main__":
    result = BuildModel().mergedData()
    #print(result)
    dnl = DeepNeuralNetwork()
    dnl.startBuildingModel(result['attention'], result['highBeta'], result['lowBeta'])
