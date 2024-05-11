import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
 
# Preprocess the data
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Create the feedforward neural network model
model = Sequential()
model.add(Dense(512, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {:.2f}%".format(accuracy * 100))

# ----------------------------------------output---------------------------------

# Epoch 1/10
# 1875/1875 [==============================] - 15s 8ms/step - loss: 0.1872 - accuracy: 0.9423 - val_loss: 0.0966 - val_accuracy: 0.9682
# Epoch 2/10
# 1875/1875 [==============================] - 14s 7ms/step - loss: 0.0807 - accuracy: 0.9748 - val_loss: 0.0891 - val_accuracy: 0.9721
# Epoch 3/10
# 1875/1875 [==============================] - 14s 8ms/step - loss: 0.0558 - accuracy: 0.9826 - val_loss: 0.0700 - val_accuracy: 0.9787
# Epoch 4/10
# 1875/1875 [==============================] - 22s 12ms/step - loss: 0.0413 - accuracy: 0.9865 - val_loss: 0.0845 - val_accuracy: 0.9751
# Epoch 5/10
# 1875/1875 [==============================] - 27s 15ms/step - loss: 0.0318 - accuracy: 0.9899 - val_loss: 0.0752 - val_accuracy: 0.9804
# Epoch 6/10
# 1875/1875 [==============================] - 32s 17ms/step - loss: 0.0279 - accuracy: 0.9905 - val_loss: 0.0768 - val_accuracy: 0.9794
# Epoch 7/10
# 1875/1875 [==============================] - 30s 16ms/step - loss: 0.0248 - accuracy: 0.9919 - val_loss: 0.0767 - val_accuracy: 0.9810
# Epoch 8/10
# 1875/1875 [==============================] - 29s 16ms/step - loss: 0.0192 - accuracy: 0.9941 - val_loss: 0.0807 - val_accuracy: 0.9828
# Epoch 9/10
# 1875/1875 [==============================] - 29s 15ms/step - loss: 0.0190 - accuracy: 0.9938 - val_loss: 0.0931 - val_accuracy: 0.9802
# Epoch 10/10
# 1875/1875 [==============================] - 28s 15ms/step - loss: 0.0150 - accuracy: 0.9952 - val_loss: 0.0948 - val_accuracy: 0.9818
# 313/313 [==============================] - 2s 7ms/step - loss: 0.0948 - accuracy: 0.9818
# Test accuracy: 98.18%