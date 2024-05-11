import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Generate some random data for training
x = np.random.rand(100, 1)
y = 2 * x + 1 + np.random.rand(100, 1)

# Create a linear regression model
model = Sequential()
model.add(Dense(1, input_shape=(1,)))

# Compile the model with mean squared error loss function and Adam optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the data
model.fit(x, y, epochs=500)

# Use the trained model to make predictions
x_test = np.array([[3.0]])
y_pred = model.predict(x_test)
print(f"Predicted value for x=3.0 is {y_pred[0][0]}")

# ----------------------------------------output---------------------------------

# Epoch 1/500
# 4/4 [==============================] - 1s 5ms/step - loss: 3.4817
# Epoch 2/500
# 4/4 [==============================] - 0s 3ms/step - loss: 3.4597
# Epoch 3/500
# 4/4 [==============================] - 0s 3ms/step - loss: 3.4377
# Epoch 4/500
# 4/4 [==============================] - 0s 4ms/step - loss: 3.4157
# Epoch 5/500
# 4/4 [==============================] - 0s 3ms/step - loss: 3.3938
# Epoch 6/500
# 4/4 [==============================] - 0s 3ms/step - loss: 3.3721
# Epoch 7/500
# 4/4 [==============================] - 0s 3ms/step - loss: 3.3501
# Epoch 8/500
# 4/4 [==============================] - 0s 3ms/step - loss: 3.3286
# Epoch 9/500
# 4/4 [==============================] - 0s 3ms/step - loss: 3.3074
# Epoch 10/500
# 4/4 [==============================] - 0s 11ms/step - loss: 3.2860
# Epoch 11/500
# 4/4 [==============================] - 0s 3ms/step - loss: 3.2648
# Epoch 12/500
# 4/4 [==============================] - 0s 3ms/step - loss: 3.2437
# Epoch 13/500
# ...
# Epoch 500/500
# 4/4 [==============================] - 0s 2ms/step - loss: 0.1101
# 1/1 [==============================] - 0s 67ms/step
# Predicted value for x=3.0 is 8.13337516784668