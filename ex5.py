import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
 
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# ----------------------------------------output---------------------------------

# Epoch 1/10
# 1875/1875 [==============================] - 43s 22ms/step - loss: 0.2017 - accuracy: 0.9390 - val_loss: 0.0419 - val_accuracy: 0.9859
# Epoch 2/10
# 1875/1875 [==============================] - 49s 26ms/step - loss: 0.0768 - accuracy: 0.9775 - val_loss: 0.0316 - val_accuracy: 0.9901
# Epoch 3/10
# 1875/1875 [==============================] - 46s 25ms/step - loss: 0.0562 - accuracy: 0.9828 - val_loss: 0.0300 - val_accuracy: 0.9904
# Epoch 4/10
# 1875/1875 [==============================] - 45s 24ms/step - loss: 0.0459 - accuracy: 0.9863 - val_loss: 0.0314 - val_accuracy: 0.9904
# Epoch 5/10
# 1875/1875 [==============================] - 35s 19ms/step - loss: 0.0375 - accuracy: 0.9883 - val_loss: 0.0287 - val_accuracy: 0.9906
# Epoch 6/10
# 1875/1875 [==============================] - 28s 15ms/step - loss: 0.0326 - accuracy: 0.9900 - val_loss: 0.0259 - val_accuracy: 0.9922
# Epoch 7/10
# 1875/1875 [==============================] - 31s 16ms/step - loss: 0.0283 - accuracy: 0.9911 - val_loss: 0.0259 - val_accuracy: 0.9925
# Epoch 8/10
# 1875/1875 [==============================] - 35s 19ms/step - loss: 0.0253 - accuracy: 0.9922 - val_loss: 0.0318 - val_accuracy: 0.9914
# Epoch 9/10
# 1875/1875 [==============================] - 30s 16ms/step - loss: 0.0230 - accuracy: 0.9929 - val_loss: 0.0273 - val_accuracy: 0.9927
# Epoch 10/10
# 1875/1875 [==============================] - 28s 15ms/step - loss: 0.0205 - accuracy: 0.9937 - val_loss: 0.0268 - val_accuracy: 0.9923

