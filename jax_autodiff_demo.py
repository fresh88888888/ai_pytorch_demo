from jax import custom_jvp
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import make_jaxpr
from jax import vmap, pmap, jit
from jax import grad, value_and_grad
from jax.test_util import check_grads


def product(x, y):
    z = x * y
    return z


x = 3.0
y = 4.0

z = product(x, y)

print(f"Input Variable x: {x}")
print(f"Input Variable y: {y}")
print(f"Product z: {z}\n")

# dz / dx
dx = grad(product, argnums=0)(x, y)
print(f"Gradient of z wrt x: {dx}")

# dz / dy
dy = grad(product, argnums=1)(x, y)
print(f"Gradient of z wrt y: {dy}")

z, dx = value_and_grad(product, argnums=0)(x, y)
print("Product z:", z)
print(f"Gradient of z wrt x: {dx}")

# Differentiating wrt first positional argument `x`
print("Differentiating wrt x")
print(make_jaxpr(grad(product, argnums=0))(x, y))

# Differentiating wrt second positional argument `y`
print("\nDifferentiating wrt y")
print(make_jaxpr(grad(product, argnums=1))(x, y))

# Modified product function. Explicity stopping the
# flow of the gradients through `y`


def product_stop_grad(x, y):
    z = x * jax.lax.stop_gradient(y)
    return z


# Differentiating wrt y. This should return 0
grad(product_stop_grad, argnums=1)(x, y)


def activate(x):
    """Applies tanh activation."""
    return jnp.tanh(x)


# Check if we can compute the gradients for a single example
grads_single_example = grad(activate)(0.5)
print("Gradient for a single input x=0.5: ", grads_single_example)


# Now we will generate a batch of random inputs, and will pass
# those inputs to our activate function. And we will also try to
# calculate the grads on the same batch in the same way as above

# Always use the PRNG
key = random.PRNGKey(1234)
x = random.normal(key=key, shape=(5,))
activations = activate(x)

print("\nTrying to compute gradients on a batch")
print("Input shape: ", x.shape)
print("Output shape: ", activations.shape)

try:
    grads_batch = grad(activate)(x)
    print("Gradients for the batch: ", grads_batch)
except Exception as ex:
    print(type(ex).__name__, ex)

grads_batch = vmap(grad(activate))(x)
print("Gradients for the batch: ", grads_batch)

jitted_grads_batch = jit(vmap(grad(activate)))

for _ in range(3):
    start_time = time.time()
    print("Gradients for the batch: ", jitted_grads_batch(x))
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print("="*50)
    print()

try:
    check_grads(jitted_grads_batch, (x,),  order=1)
    print("Gradient match with gradient calculated using finite differences")
except Exception as ex:
    print(type(ex).__name__, ex)

x = 0.5

print("First order derivative: ", grad(activate)(x))
print("Second order derivative: ", grad(grad(activate))(x))
print("Third order derivative: ", grad(grad(grad(activate)))(x))

# An example of a mathematical operation in your workflow


def log1pexp(x):
    """Implements log(1 + exp(x))"""
    return jnp.log(1. + jnp.exp(x))


# This works fine
print("Gradients for a small value of x: ", grad(log1pexp)(5.0))

# But what about for very large values of x for which the
# exponent operation will explode
print("Gradients for a large value of x: ", grad(log1pexp)(500.0))


@custom_jvp
def log1pexp(x):
    """Implements log(1 + exp(x))"""
    return jnp.log(1. + jnp.exp(x))


@log1pexp.defjvp
def log1pexp_jvp(primals, tangents):
    """Tells JAX to differentiate the function in the way we want."""
    x, = primals
    x_dot, = tangents
    ans = log1pexp(x)
    # This is where we define the correct way to compute gradients
    ans_dot = (1 - 1/(1 + jnp.exp(x))) * x_dot
    return ans, ans_dot


# Let's now compute the gradients for large values
print("Gradients for a small value of x: ", grad(log1pexp)(500.0))

# What about the Jaxpr?
print(make_jaxpr(grad(log1pexp))(500.0))
