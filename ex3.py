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

# Create the perceptron model
model = Sequential()
model.add(Dense(10, input_dim=x_train.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test accuracy: {:.2f}%".format(accuracy * 100))

# ----------------------------------------output---------------------------------

# 11490434/11490434 [==============================] - 7s 1us/step
# Epoch 1/10
# 1875/1875 [==============================] - 5s 2ms/step - loss: 0.4740 - accuracy: 0.8773 - val_loss: 0.3090 - val_accuracy: 0.9171
# Epoch 2/10
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.3036 - accuracy: 0.9152 - val_loss: 0.2782 - val_accuracy: 0.9222
# Epoch 3/10
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.2837 - accuracy: 0.9204 - val_loss: 0.2735 - val_accuracy: 0.9238
# Epoch 4/10
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.2733 - accuracy: 0.9244 - val_loss: 0.2687 - val_accuracy: 0.9242
# Epoch 5/10
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.2665 - accuracy: 0.9261 - val_loss: 0.2689 - val_accuracy: 0.9262
# Epoch 6/10
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.2616 - accuracy: 0.9275 - val_loss: 0.2668 - val_accuracy: 0.9261
# Epoch 7/10
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.2586 - accuracy: 0.9289 - val_loss: 0.2645 - val_accuracy: 0.9273
# Epoch 8/10
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.2553 - accuracy: 0.9295 - val_loss: 0.2671 - val_accuracy: 0.9263
# Epoch 9/10
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.2528 - accuracy: 0.9302 - val_loss: 0.2641 - val_accuracy: 0.9256
# Epoch 10/10
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.2508 - accuracy: 0.9309 - val_loss: 0.2688 - val_accuracy: 0.9263
# 313/313 [==============================] - 1s 2ms/step - loss: 0.2688 - accuracy: 0.9263
# Test accuracy: 92.63%