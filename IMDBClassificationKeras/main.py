import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM,Dense,MaxPooling1D,Conv1D
from keras.layers.embeddings import Embedding
from keras import regularizers

top_words = 5000
max_seq_len = 500
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
X_train = sequence.pad_sequences(X_train, maxlen=max_seq_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_seq_len)


model = Sequential()
model.add(Embedding(top_words,64,input_length=max_seq_len, embeddings_regularizer=regularizers.l1(0.001)))
model.add(LSTM(128))
model.add(Dense(32,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss= keras.losses.binary_crossentropy,
			  optimizer= keras.optimizers.Adam(lr=0.0001),
			  metrics=['accuracy'])
model.fit(X_train, y_train,
		  batch_size=32,
		  epochs = 10,
		  verbose = 1,
		  validation_data=(X_test,y_test))
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])