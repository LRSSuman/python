import tensorflow as tf

# Define two vectors as constants
vector1 = tf.constant([1, 2, 3])
vector2 = tf.constant([4, 5, 6])

# Add the vectors
result = tf.add(vector1, vector2)

# Print the result
print(result)

# ----------------------------------------output---------------------------------

# tf.Tensor([5 7 9], shape=(3,), dtype=int32)