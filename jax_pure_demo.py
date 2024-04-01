from joblib import Parallel, delayed
import jax
import timeit
import time
import numpy as np
import jax.numpy as jnp
from jax import grad
from jax import jit
from jax import lax
from jax import random

# A global variable
counter = 5


def add_global_value(x):
    """
    A function that relies on the global variable `counter` for
    doing some computation.
    """
    return x + counter


x = 2
# We will `JIT` the function so that it runs as a JAX transformed
# function and not like a normal python function
y = jit(add_global_value)(x)
print("Global variable value: ", counter)
print(f"First call to the function with input {x} with global variable value {counter} returned {y}")

# Someone updated the global variable value later in the code
counter = 10

# Call the function again
y = jit(add_global_value)(x)
print("\nGlobal variable changed value: ", counter)
print(f"Second call to the function with input {x} with global variable value {counter} returned {y}")

# Change the type of the argument passed to the function
# In this case we will change int to float (2 -> 2.0)
x = 2.0
y = jit(add_global_value)(x)
print(f"Third call to the function with input {x} with global variable value {counter} returned {y}")

# Change the shape of the argument
x = jnp.array([2])

# Changing global variable value again
counter = 15

# Call the function again
y = jit(add_global_value)(x)
print(f"Third call to the function with input {x} with global variable value {counter} returned {y}")


def apply_sin_to_global():
    return jnp.sin(jnp.pi / counter)


y = apply_sin_to_global()
print("Global variable value: ", counter)
print(f"First call to the function with global variable value {counter} returned {y}")


# Change the global value again
counter = 90
y = apply_sin_to_global()
print("\nGlobal variable value: ", counter)
print(f"Second call to the function with global variable value {counter} returned {y}")

# A function that takes an actual array object
# and add all the elements present in it


def add_elements(array, start, end, initial_value=0):
    res = 0

    def loop_fn(i, val):
        return val + array[i]
    return lax.fori_loop(start, end, loop_fn, initial_value)


# Define an array object
array = jnp.arange(5)
print("Array: ", array)
print("Adding all the array elements gives: ", add_elements(array, 0, len(array), 0))


# Redefining the same function but this time it takes an
# iterator object as an input
def add_elements(iterator, start, end, initial_value=0):
    res = 0

    def loop_fn(i, val):
        return val + next(iterator)
    return lax.fori_loop(start, end, loop_fn, initial_value)


# Create an iterator object
iterator = iter(np.arange(5))
print("\nIterator: ", iterator)
print("Adding all the elements gives: ", add_elements(iterator, 0, 5, 0))


def return_as_it_is(x):
    """Returns the same element doing nothing. A function that isn't
    using `globals` or any `iterator`
    """
    print(f"I have received the value")
    return x


# First call to the function
print(f"Value returned on first call: {jit(return_as_it_is)(2)}\n")

# Second call to the fucntion with different value
print(f"Value returned on second call: {jit(return_as_it_is)(4)}")

# Function that uses stateful objects but internally and is still pure


def pure_function_with_stateful_obejcts(array):
    array_dict = {}
    for i in range(len(array)):
        array_dict[i] = array[i] + 10
    return array_dict


array = jnp.arange(5)

# First call to the function
print(f"Value returned on first call: {jit(pure_function_with_stateful_obejcts)(array)}")

# Second call to the fucntion with different value
print(f"\nValue returned on second call: {jit(pure_function_with_stateful_obejcts)(array)}")

# If I set the seed, would I get the same sequence of random numbers every time?

for i in range(10):
    # Set initial value by providing a seed value
    seed = 0
    np.random.seed(seed)

    # Generate a random integer from a range of [0, 5)
    random_number = np.random.randint(0, 5)
    print(f"Seed: {seed} -> Random number generated: {random_number}")

# Array of 10 values
array = np.arange(10)

for i in range(5):
    # Set initial value by providing a seed value
    seed = 1234
    # np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Choose array1 and array2 indices
    array_1_idx = rng.choice(array, size=8)
    array_2_idx = rng.choice(array, size=2)

    # Split the array into two sets
    array_1 = array[array_1_idx]
    array_2 = array[array_2_idx]

    print(f"Iteration: {i+1}  Seed value: {seed}\n")
    print(f"First array: {array_1}  Second array: {array_2}")
    print("="*50)
    print("")


def get_sequence(seed, size=5):
    rng = np.random.default_rng(seed)
    array = np.arange(10)
    return rng.choice(array, size=size)


# Instantiate SeedSequence
seed = 1234
ss = np.random.SeedSequence(seed)

# Spawn 2 child seed sequence
child_seeds = ss.spawn(2)

# Run the function a few times in parallel to check if we get
# same RNG sequence
for i in range(5):
    res = []
    for child_seed in child_seeds:
        res.append(delayed(get_sequence)(child_seed))
    res = Parallel(n_jobs=2)(res)
    print(f"Iteration: {i+1} Sequences: {res}")
    print("="*70)

# Global seed
np.random.seed(1234)


def A():
    return np.random.choice(["a", "A"])


def B():
    return np.random.choice(["b", "B"])


for i in range(2):
    C = A() + B()
    print(f"Iteration: {i+1}  C: {C}")

# Define a state
seed = 1234
key = random.PRNGKey(1234)
print(key)

# Passing the original key to a random function
random_integers = random.randint(key=key, minval=0, maxval=10, shape=[5])
print(random_integers)

# What if we want to call another function?
# Don't use the same key. Split the original key, and then pass it
print("Original key: ", key)

# Split the key. By default the number of splits is set to 2
# You can specify explicitly how many splits you want to do
key, subkey = random.split(key, num=2)

print("New key: ",  key)
print("Subkey: ", subkey)

# Call another random function with the new key
random_floats = random.normal(key=key, shape=(5,), dtype=jnp.float32)
print(random_floats)

for i in range(3):
    key = random.PRNGKey(1234)
    print(f"Iteration: {i+1}\n")
    print(f"Original key: {key}")
    key, subkey = random.split(key)
    print(f"First subkey: {key}")
    print(f"Second subkey: {subkey}")
    print("="*50)
    print("")

key = random.PRNGKey(111)
print(f"Original key: {key}\n")

subkeys = random.split(key, num=5)

for i, subkey in enumerate(subkeys):
    print(f"Subkey no: {i+1}  Subkey: {subkey}")

key = random.PRNGKey(1234)
random_integers_1 = random.randint(key=key, minval=0, maxval=10, shape=(5,))

key = random.PRNGKey(1234)
key, *subkeys = random.split(key, 5)
random_integers_2 = []

for subkey in subkeys:
    num = random.randint(key=subkey, minval=0, maxval=10, shape=(1,))
    random_integers_2.append(num)

random_integers_2 = np.stack(random_integers_2, axis=-1)[0]

print("Generated all at once: ", random_integers_1)
print("Generated sequentially: ", random_integers_2)


def sampler1(key):
    return random.uniform(key=key, minval=0, maxval=1, shape=(2,))


def sampler2(key):
    return 2 * random.uniform(key=key, minval=0, maxval=1, shape=(2,))


key = random.PRNGKey(0)
sample_1 = sampler1(key=key)
sample_2 = sampler2(key=key)

print("First sample: ", sample_1)
print("Second sample: ", sample_2)


def sampler1():
    return np.random.uniform(low=0, high=1, size=(2,))


def sampler2():
    return 2 * np.random.uniform(low=0, high=1, size=(2,))


np.random.seed(0)
sample_1 = sampler1()
sample_2 = sampler2()

print("First sample: ", sample_1)
print("Second sample: ", sample_2)


def apply_activation(x):
    return jnp.maximum(0.0, x)


def get_dot_product(W, X):
    return jnp.dot(W, X)


# Always use a seed
key = random.PRNGKey(1234)
W = random.normal(key=key, shape=[1000, 10000], dtype=jnp.float32)

# Never reuse the key
key, subkey = random.split(key)
X = random.normal(key=subkey, shape=[10000, 20000], dtype=jnp.float32)

# JIT the functions we have
dot_product_jit = jit(get_dot_product)
activation_jit = jit(apply_activation)

for i in range(3):
    start = time.time()
    # Don't forget to use `block_until_ready(..)`
    # else you will be recording dispatch time only
    Z = dot_product_jit(W, X).block_until_ready()
    end = time.time()
    print(f"Iteration: {i+1}")
    print(f"Time taken to execute dot product: {end - start:.2f} seconds", end="")

    start = time.time()
    A = activation_jit(Z).block_until_ready()
    print(f", activation function: {time.time()-start:.2f} seconds")

# Make jaxpr for the activation function
print(jax.make_jaxpr(activation_jit)(Z))

# Make jaxpr for the activation function
print(jax.make_jaxpr(dot_product_jit)(W, X))

# We know that `print` introduces but impurity but it is
# also very useful to print values while debugging. How does
# jaxprs interpret that?


def number_squared(num):
    print("Received: ", num)
    return num ** 2


# Compiled version
number_squared_jit = jit(number_squared)

# Make jaxprs
print(jax.make_jaxpr(number_squared_jit)(2))

# Subsequent calls to the jitted function
for i, num in enumerate([2, 4, 8]):
    print("Iteration: ", i+1)
    print("Result: ", number_squared_jit(num))
    print("="*50)

squared_numbers = []

# An impure function (using a global state)


def number_squared(num):
    global squared_numbers
    squared = num ** 2
    squared_numbers.append(squared)
    return squared


# Compiled verison
number_squared_jit = jit(number_squared)

# Make jaxpr
print(jax.make_jaxpr(number_squared_jit)(2))


# Calling the two functions into a single function
# so that we can jit this function instead of jitting them
def forward_pass(W, X):
    Z = get_dot_product(W, X)
    A = apply_activation(Z)
    return Z, A


# Always use a seed
key = random.PRNGKey(1234)

# We will use much bigger array this time
W = random.normal(key=key, shape=[2000, 10000], dtype=jnp.float32)

# Never reuse the key
key, subkey = random.split(key)
X = random.normal(key=subkey, shape=[10000, 20000], dtype=jnp.float32)

# JIT the functions we have individually
dot_product_jit = jit(get_dot_product)
activation_jit = jit(apply_activation)

# JIT the function that wraps both the functions
forward_pass_jit = jit(forward_pass)

for i in range(3):
    start = time.time()
    # Don't forget to use `block_until_ready(..)`
    # else you will be recording dispatch time only
    Z = dot_product_jit(W, X).block_until_ready()
    end = time.time()
    print(f"Iteration: {i+1}")
    print(f"Time taken to execute dot product: {end - start:.2f} seconds", end="")

    start = time.time()
    A = activation_jit(Z).block_until_ready()
    print(f", activation function: {time.time()- start:.2f} seconds")

    # Now measure the time with a single jitted function that calls
    # the other two functions
    Z, A = forward_pass_jit(W, X)
    Z, A = Z.block_until_ready(), A.block_until_ready()
    print(f"Time taken by the forward pass function: {time.time()- start:.2f} seconds")
    print("")
    print("="*50)


# def multiply_n_times(x, n):
#     count = 0
#     res = 1
#     while count < n:
#         res = res * x
#         count += 1
#     return x


# try:
#     val = jit(multiply_n_times)(2, 5)
# except Exception as ex:
#     print(type(ex).__name__, ex)

# Jitting the expensive computational part
def multiply(x, i):
    return x * i


# Specifying the static args
multiply_jit = jit(multiply, static_argnums=0)

# Leaving it as it as


def multiply_n_times(x, n):
    count = 0
    res = 1
    while count < n:
        res = multiply_jit(x, res)
        count += 1
    return res


multiply_n_times_timeit = timeit.timeit(stmt='multiply_n_times(2, 5)', setup='from __main__ import multiply_n_times')
print('multiply_n_times_timeit: ', multiply_n_times_timeit)
