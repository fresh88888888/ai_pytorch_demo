from jax.tree_util import register_pytree_node_class
from jax.tree_util import register_pytree_node
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from jax import make_jaxpr
from jax import vmap, pmap, jit
from jax import grad, value_and_grad
from jax.test_util import check_grads

# A list as a pytree
example_1 = [1, 2, 3]

# As in normal Python code, a list that represents pytree
# can contain obejcts of any type
example_2 = [1, 2, "a", "b", (3, 4)]

# Similarly we can define pytree using a tuple as well
example_3 = (1, 2, "a", "b", (3, 4))

# We can define the same pytree using a dict as well
example_4 = {"k1": 1, "k2": 2, "k3": "a", "k4": "b", "k5": (3, 4)}

# Let's check the number of leaves and the corresponding values in the above pytrees
example_pytrees = [example_1, example_2, example_3, example_4]
for pytree in example_pytrees:
    leaves = jax.tree_leaves(pytree)
    print(f"Pytree: {repr(pytree):<30}")
    print(f"Number of leaves: {len(leaves)}")
    print(f"Leaves are: {leaves}\n")
    print("="*50)


# Check if we can make a pytree from a DeviceArray
example_5 = jnp.array([1, 2, 3])
leaves = jax.tree_leaves(example_5)
print(f"DeviceArray: {repr(example_5):<30}")
print(f"Number of leaves: {len(leaves)}")
print(f"Leaves are: {leaves}")

# We will use the `example_2` pytree for this purpose.
# Our pytree looks like this: [1, 2, 'a', 'b', (3, 4)]
# We will unflatten it, obtain the leaves, and the tree structure as well

example_2_leaves, example_2_treedef = jax.tree_flatten(example_2)
print(f"Original Pytree: {repr(example_2)}")
print(f"Leaves: {repr(example_2_leaves)}")
print(f"Pytree structure: {repr(example_2_treedef)}")


def change_even_positioned_leaf(x, pos):
    if pos % 2 == 0:
        return x * 2
    else:
        return x


transformed_leaves = [
    change_even_positioned_leaf(leaf, pos+1) for pos, leaf in enumerate(example_2_leaves)
]

print(f"Original leaves:    {repr(example_2_leaves)}")
print(f"Transformed leaves: {repr(transformed_leaves)}")


class Counter:
    def __init__(self, count, name):
        self.count = count
        self.name = name

    def __repr__(self):
        return f"Counter value = {self.count}"

    def increment(self):
        return self.count + 1

    def decrement(self):
        return self.count - 1


# Because JAX doesn't know how to flattent and unflatten these custom objects
# hence we need to define those methods for these objects

def flatten_counter(tree):
    """Specifies how to flatten a Counter class object.

    Args:
        tree: Counter class object represented as Pytree node
    Returns:
        A pair of an iterable with the children to be flattened recursively,
        and some opaque auxiliary data to pass back to the unflattening recipe.
        The auxiliary data is stored in the treedef for use during unflattening.
        The auxiliary data could be used, e.g., for dictionary keys.
    """

    children = (tree.count,)
    aux_data = tree.name  # We don't want to treat the name as a child
    return (children, aux_data)


def unflatten_counter(aux_data, children):
    """Specifies how to unflattening a Counter class object.

    Args:
        aux_data: the opaque data that was specified during flattening of the
            current treedef.
        children: the unflattened children
    Returns:
        A re-constructed object of the registered type, using the specified
        children and auxiliary data.
    """
    return Counter(*children, aux_data)


# Now all we need to do is to tell JAX that we need to Register our class as
# a Pytree node and it need to treat all the objects of that class as such
register_pytree_node(
    Counter,
    flatten_counter,    # tell JAX what are the children nodes
    unflatten_counter   # tell JAX how to pack back into a `Counter`
)

# An instance of the Counter class
my_counter = Counter(count=5, name="Counter_class_as_pytree_node")

# Flatten the custom object
my_counter_leaves, my_counter_treedef = jax.tree_flatten(my_counter)

# Unflatten
my_counter_reconstructed = jax.tree_unflatten(
    treedef=my_counter_treedef, leaves=my_counter_leaves
)
print(f"Original Pytree: {repr(my_counter)}")
print(f"Leaves: {repr(my_counter_leaves)}")
print(f"Pytree structure: {repr(my_counter_treedef)}")
print(f"Reconstructed Pytree: {repr(my_counter_reconstructed)}")

# Another instance
my_counter_2 = Counter(count=5, name="Counter_class_as_pytree_node")

# Flatten the custom object
my_counter_2_leaves, my_counter_2_treedef = jax.tree_flatten(my_counter)

# Check if the treedef are same for both the pytrees
print(my_counter_treedef == my_counter_2_treedef)


def activate(x):
    """Applies tanh activation."""
    return jnp.tanh(x["weights"])


# Always use the PRNG
key = random.PRNGKey(1234)
example_pytree = {"weights": random.normal(key=key, shape=(5,))}

# We will now use vmap and grad to compute the gradients per sample
grads_example_pytree = vmap(grad(activate), in_axes=({"weights": 0},))(example_pytree)

print("Original pytree:")
print(f" {repr(example_pytree)}\n")
print("Leaves in the pytree:")
print(f"{repr(jax.tree_leaves(example_pytree))}\n")
print("Gradients per example:")
print(f"{grads_example_pytree}\n")

key = random.PRNGKey(111)
key, subkey = random.split(key)

# Generate some random data
x = random.normal(key=key, shape=(128, 1))
# Let's just do y = 10x + 20
y = 10 * x + 20

# plt.plot(x, y, marker='x', label='Generated linear function')
# plt.legend()
# plt.show()


def initialize_params(key, dims):
    """Initialize the weights and biases of the MLP.

    Args:
        key: PRNG key
        dims: List of integers
    Returns:
        A pytree of initialized paramters for each layer
    """

    params = []

    for dim_in, dim_out in zip(dims[:-1], dims[1:]):
        key, subkey = random.split(key)
        weights = random.normal(key=key, shape=(dim_in, dim_out)) * jnp.sqrt(2 / dim_in)
        biases = jnp.zeros(shape=(dim_out))
        params.append({"weights": weights, "biases": biases})

    return params


# Initialize the parameters
params = initialize_params(key=subkey, dims=[1, 128, 128, 1])

# We can inspect the shape of the intialized params as well
shapes = jax.tree_map(lambda layer_params: layer_params.shape, params)

for i, shape in enumerate(shapes):
    print(f"Layer {i+1} => Params shape: {shape}")


def forward(params, x):
    """Forward pass for the MLP

    Args:
        params: A pytree containing the parameters of the network
        x: Inputs
    """
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(x @ layer['weights'] + layer['biases'])
    return x @ last['weights'] + last['biases']


def loss_fn(params, x, y):
    """Mean squared error loss function."""
    return jnp.mean((forward(params, x) - y) ** 2)


@jax.jit
def update(params, x, y):
    """Updates the parameters of the network.

    Args:
        params: A pytree containing the parameters of the network
        x : Inputs
        y:  Outputs
    Returns:
        Pytree with updated values
    """

    # 1. Calculate the gradients based on the loss
    grads = jax.grad(loss_fn)(params, x, y)

    # 2. Update the parameters using `tree_multi_map(...)`
    return jax.tree_map(lambda p, g: p - 0.0001 * g, params, grads)


# Run the model for a few iterations
for _ in range(200):
    params = update(params, x, y)


# Plot the predictions and the ground truth
plt.plot(x, y, marker='x', label='Generated linear function')
plt.plot(x, forward(params, x), marker="x", label="Predictions")
plt.legend()
plt.show()
