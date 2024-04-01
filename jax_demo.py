import time
import jax
import numpy as np
import jax.numpy as jnp


# We will create two arrays, one with numpy and other with jax
# to check the common things and the differences
array_numpy = np.arange(10, dtype=np.int32)
array_jax = jnp.arange(10, dtype=jnp.int32)

print("Array created using numpy: ", array_numpy)
print("Array created using JAX: ", array_jax)

# What types of array are these?
print(f"array_numpy is of type : {type(array_numpy)}")
print(f"array_jax is of type : {type(array_jax)}")

# Find the max element. Similarly you can find `min` as well
print(f"Maximum element in ndarray: {array_numpy.max()}")
print(f"Maximum element in DeviceArray: {array_jax.max()}")

# Reshaping
print("Original shape of ndarray: ", array_numpy.shape)
print("Original shape of DeviceArray: ", array_jax.shape)

array_numpy = array_numpy.reshape(-1, 1)
array_jax = array_jax.reshape(-1, 1)

print("\nNew shape of ndarray: ", array_numpy.shape)
print("New shape of DeviceArray: ", array_jax.shape)

# Absoulte pairwise difference
print("Absoulte pairwise difference in ndarray")
print(np.abs(array_numpy - array_numpy.T))

print("\nAbsoulte pairwise difference in DeviceArray")
print(jnp.abs(array_jax - array_jax.T))

# Are they equal?
print("\nAre all the values same?", end=" ")
print(jnp.all(np.abs(array_numpy - array_numpy.T) == jnp.abs(array_jax - array_jax.T)))

array1 = np.arange(5, dtype=np.int32)
array2 = jnp.arange(5, dtype=jnp.int32)

print("Original ndarray: ", array1)
print("Original DeviceArray: ", array2)

# Item assignment
array1[4] = 10
print("\nModified ndarray: ", array1)
print("\nTrying to modify DeviceArray-> ", end=" ")

try:
    array2[4] = 10
    print("Modified DeviceArray: ", array2)
except Exception as ex:
    print(type(ex).__name__, ex)


# Modifying DeviceArray elements at specific index/indices
array2_modified = array2.at[4].set(10)

# Equivalent => array2_modified = jax.ops.index_update(array2, 4, 10)
print("Original DeviceArray: ", array2)
print("Modified DeviceArray: ", array2_modified)

# Of course, updates come in many forms!
print(array2.at[4].add(6))
print(array2.at[4].max(20))
print(array2.at[4].min(-1))

# Create two random arrays sampled from a uniform distribution
array1 = np.random.uniform(size=(8000, 8000)).astype(np.float32)
array2 = jax.random.uniform(jax.random.PRNGKey(0), (8000, 8000), dtype=jnp.float32)  # More on PRNGKey later!
print("Shape of ndarray: ", array1.shape)
print("Shape of DeviceArray: ", array2.shape)

# # Dot product on ndarray
# start_time = time.time()
# res = np.dot(array1, array1)
# print(f"Time taken by dot product op on ndarrays: {time.time()-start_time:.2f} seconds")

# # Dot product on DeviceArray
# start_time = time.time()
# res = jnp.dot(array2, array2)
# print(f"Time taken by dot product op on DeviceArrays: {time.time()-start_time:.2f} seconds")

# First we will time it by converting the computation results to ndarray
# start_time = time.time()
# np.asarray(jnp.dot(array2, array2))
# print(f"Time taken by dot product op on ndarrays: {time.time()-start_time:.2f} seconds")

# start_time = time.time()
# jnp.dot(array2, array2).block_until_ready()
# print(f"Time taken by dot product op on DeviceArrays: {time.time()-start_time:.2f} seconds")

print("Types promotion in numpy =>", end=" ")
print((np.int8(32) + 4).dtype)

print("Types promtoion in JAX =>", end=" ")
print((jnp.int8(32) + 4).dtype)

# Types promotion in numpy => int64
# Types promtoion in JAX => int8

array1 = np.random.randint(5, size=(2), dtype=np.int32)
print("Implicit numpy casting gives: ", (array1 + 5.0).dtype)

# Check the difference in semantics of the above function in JAX
array2 = jax.random.randint(jax.random.PRNGKey(0),
                            minval=0,
                            maxval=5,
                            shape=[2],
                            dtype=jnp.int32
                            )
print("Implicit JAX casting gives: ", (array2 + 5.0).dtype)


def squared(x):
    return x**2


x = 4.0
y = squared(x)

dydx = jax.grad(squared)
print("First order gradients of y wrt x: ", dydx(x))
print("Second order gradients of y wrt x: ", jax.grad(dydx)(x))
