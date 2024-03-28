import time
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
