import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
import os
import scipy as sc
import h5py
import numpy as np 
import matplotlib.pyplot as plt


mat = h5py.File("../../Datasets/happy-house-dataset/train_happy.h5",mode='r') 
print(list(mat.keys()))
classes = np.array(mat.get("list_classes"))
x_train = np.array(mat.get("train_set_x"))
y_train = np.array(mat.get("train_set_y"))
mat = h5py.File("../../Datasets/happy-house-dataset/test_happy.h5",mode='r') 
print(list(mat.keys()))
x_test = np.array(mat.get("test_set_x"))
y_test = np.array(mat.get("test_set_y"))

x_train = 0.07*x_train[:,:,:,0] + 0.72*x_train[:,:,:,1] + 0.21*x_train[:,:,:,2]
x_test = 0.07*x_test[:,:,:,0] + 0.72*x_test[:,:,:,1] + 0.21*x_test[:,:,:,2]
x_train=np.reshape(x_train, (600,64,64,1))
x_test=np.reshape(x_test, (x_test.shape[0],64,64,1))
print(x_test.shape)
# plt.imshow(x_train[1])
# plt.show()
# plt.imshow(train_inputs[12])
# plt.show()
# plt.imshow(train_inputs[15])
# plt.show()


model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(64,64,1), kernel_regularizer=regularizers.l1(0.001)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l1_l2(0.001,0.001)))
model.add(Dense(1,activation="sigmoid", kernel_regularizer=regularizers.l1_l2(0.001,0.001)))


# model.compile(loss=keras.losses.binary_crossentropy,
#               optimizer=keras.optimizers.Adam(lr=0.0001),
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=32,
#           epochs=100,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

#model.save_weights("model2.h5")
model.load_weights("model2.h5")
y = model.predict_classes(x_train[0:3])
print(y)
print(y_train[0:3])