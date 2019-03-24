from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

print(tf.__version__)
tf.enable_eager_execution()

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)

tf.convert_to_tensor(X_train, dtype=tf.float32)
tf.convert_to_tensor(X_test, dtype=tf.float32)
tf.convert_to_tensor(y_train, dtype=tf.float32)
tf.convert_to_tensor(y_test, dtype=tf.float32)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=X_train.shape),
    keras.layers.Dense(40, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Shape: ", X_train.shape)
print("Shape: ", y_train.shape)

model.fit(X_train, y_train, epochs=50)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

test = [[[5697,958,10192,213943,3255,1942,12090,42660,80,75]]]
predictions = model.predict(test)
print(predictions)