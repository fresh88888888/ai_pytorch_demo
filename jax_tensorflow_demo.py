import numpy as np
import tensorflow as tf
import keras
from keras import layers, callbacks

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

# We can convert any tensor object to `ndarray` by calling the `numpy()` method
y = tf.constant([1, 2, 3], dtype=tf.int8).numpy()
print(f"`y` is now a {type(y)} object and have a value == {y}")

# tf.constant(..) is no special. Let's create a tensor using a diff method
x = tf.ones(2, dtype=tf.int8)
print(x)

try:
    x[0] = 3
except Exception as ex:
    print("\n", type(ex).__name__, ex)

# Check all the properties of a tensor object
print(f"Shape of x : {x.shape}")
print(f"Another method to obtain the shape using `tf.shape(..)`: {tf.shape(x)}")
print(f"\nRank of the tensor: {x.ndim}")
print(f"dtype of the tensor: {x.dtype}")
print(f"Total size of the tensor: {tf.size(x)}")
print(f"Values of the tensor: {x.numpy()}")

# Create a tensor first. Here is another way
x = tf.cast([1, 2, 3, 4, 5], dtype=tf.float32)
print("Original tensor: ", x)

mask = x % 2 == 0
print("Original mask: ", mask)

mask = tf.cast(mask, dtype=x.dtype)
print("Mask casted to original tensor type: ", mask)

# Some kind of operation on an tensor that is of same size
# or broadcastable to the original tensor. Here we will simply
# use the range object to create that tensor
temp = tf.cast(tf.range(1, 6) * 100, dtype=x.dtype)

# Output tensor
# Input tensor -> [1, 2, 3, 4, 5]
# Mask -> [0, 1, 0, 1, 0]
out = x * (1-mask) + mask * temp
print("Output tensor: ", out)

# Another way to achieve the same thing
indices_to_update = tf.where(x % 2 == 0)
print("Indices to update: ", indices_to_update)

# Update the tensor values
updates = [200., 400.]
out = tf.tensor_scatter_nd_update(x, indices_to_update, updates)
print("\nOutput tensor")
print(out)

# This works!
arr = np.random.randint(5, size=(5,), dtype=np.int32)
print("Numpy array: ", arr)

print("Accessing numpy array elements based on a  condition with irregular strides",
      arr[[1, 4]])
# try:
#     print(
#         "Accessing tensor elements based on a  condition with irregular strides", x[[1, 4]])
# except Exception as ex:
#     print(type(ex).__name__, ex)

print("Original tensor: ", x.numpy())

# Using the indices that we used for mask
print("\nIndices to update: ", indices_to_update.numpy())

print("\n Accesing tensor elements using gather")
print("\n", tf.gather(x, indices_to_update).numpy())

#  An example with a python list
y = tf.convert_to_tensor([1, 2, 3])
print("Tensor from python list: ", y)

#  An example with a ndarray
y = tf.convert_to_tensor(np.array([1, 2, 3]))
print("Tensor from ndarray: ", y)

#  An example with symbolic tensors
with tf.compat.v1.Graph().as_default():
    y = tf.convert_to_tensor(tf.compat.v1.placeholder(
        shape=[None, None, None], dtype=tf.int32))
print("Tensor from python list: ", y)

