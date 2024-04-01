import os
import glob
import time
import cv2
import urllib
import requests
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from jax import make_jaxpr
from jax import grad, vmap, pmap, jit


def dot_product(array1, array2):
    """Performs dot product on two jax arrays."""
    return jnp.dot(array1, array2)


def print_results(array1, array2, res, title=""):
    """Utility to print arrays and results"""
    if title:
        print(title)
        print("")
    print("First array => Shape: ", array1.shape)
    print(array1)
    print("")
    print("Second array => Shape: ", array2.shape)
    print(array2)
    print("")
    print("Results => Shape: ", res.shape)
    print(res)


array1 = jnp.array([1, 2, 3, 4])
array2 = jnp.array([5, 6, 7, 8])
res = dot_product(array1, array2)

print_results(array1, array2, res, title="Dot product of two vectors")


# What if we want to do this for a batch of vectors?
array1 = jnp.stack([jnp.array([1, 2, 3, 4]) for i in range(5)])
array2 = jnp.stack([jnp.array([5, 6, 7, 8]) for i in range(5)])

# First way to do batch vector product using loops
res1 = []
for i in range(5):
    res1.append(dot_product(array1[i], array2[i]))
res1 = jnp.stack(res1)


# In numpy, we can use `einsum` for the same
res2 = np.einsum('ij,ij-> i', array1, array2)

# We can even simplify einsum and chain two oprations to
# achieve the same
res3 = np.sum(array1*array2, axis=1)

# Let's check the results
print_results(array1,
              array2,
              res1,
              title="1. Dot product on a batch of vectors using loop")
print("="*70, "\n")
print_results(array1,
              array2,
              res2,
              title="2. Dot product on a batch of vectors in numpy using einsum")
print("="*70, "\n")
print_results(array1,
              array2,
              res3,
              title="3. Dot product on a batch of vectors using elementwise multiplication and sum")

# Transform the `dot_product` function defined above
# using the `vmap` transformation
batch_dot_product = vmap(dot_product, in_axes=(0, 0))

# Using vmap transformed function
res4 = batch_dot_product(array1, array2)
print_results(array1,
              array2,
              res4,
              title="Dot product of a batch of vectors using vmap")

# A vector
array1 = jnp.array([1, 2, 3, 4])

# We have a batch of vectors as well already `array2` which looks like this
# [[5 6 7 8]
# [5 6 7 8]
# [5 6 7 8]
# [5 6 7 8]
# [5 6 7 8]]

# We will now perform the dot product of array1 (a single vetor) with a batch
# of vectors (array2 in this case). We will pass `None` in the `in_axes(..)` argument
# to say that the first input doesn't have a batch dimension
res5 = vmap(dot_product, in_axes=(None, 0))(array1, array2)
print_results(array1, array2, res5, title="Only one of the inputs in batched")

# Like JIT, you can inpsect the transformation using jaxprs
print(make_jaxpr(vmap(dot_product, in_axes=(None, 0)))(array1, array2))


def download_images():
    # urllib.request.urlretrieve("https://i.imgur.com/Bvro0YD.png", "img/elephant.png")
    # urllib.request.urlretrieve("https://images-eu.ssl-images-amazon.com/images/I/A1WuED4KiRL.jpg", "img/cat.jpg")
    # urllib.request.urlretrieve("https://upload.wikimedia.org/wikipedia/commons/1/18/Dog_Breeds.jpg", "img/dog.jpg")
    # urllib.request.urlretrieve(
    #     "https://upload.wikimedia.org/wikipedia/commons/1/1e/The_Korean_Lucky_Bird_%28182632069%29.jpeg", "img/bird.jpg")
    # urllib.request.urlretrieve(
    #     "https://upload.wikimedia.org/wikipedia/commons/e/ea/Vervet_Monkey_%28Chlorocebus_pygerythrus%29.jpg", "img/monkey.jpg")
    # urllib.request.urlretrieve("https://upload.wikimedia.org/wikipedia/commons/f/fa/Puppy.JPG", "img/puppy.jpg")
    # urllib.request.urlretrieve("https://upload.wikimedia.org/wikipedia/commons/2/2c/Lion-1.jpg", "img/lion.jpg")
    # urllib.request.urlretrieve(
    #     "https://upload.wikimedia.org/wikipedia/commons/4/41/Siberischer_tiger_de_edit02.jpg", "img/tiger.jpg")
    print("Downloading finished")


# Download the images
download_images()


def read_images(size=(800, 800)):
    """Read jpg/png images from the disk.

    Args:
        size: Size to be used while resizing
    Returns:
        A JAX array of images
    """
    png_images = sorted(glob.glob("img/*.png"))
    jpg_images = sorted(glob.glob("img/*.jpg"))
    all_images = png_images + jpg_images

    images = []

    for img in all_images:
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        images.append(img)

    return jnp.array(images)


# Read and resize
images = read_images()
print("Total number of images: ", len(images))

# Utility function for plotting the images


def plot_images(images, batch_size, num_cols=4, figsize=(15, 8), title="Images "):
    num_rows = batch_size // num_cols

    _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i, img in enumerate(images):
        ax[i // num_cols, i % num_cols].imshow(images[i])
        ax[i // num_cols, i % num_cols].axis("off")
        # ax[i // num_cols, i % num_cols].set_title(str(i+1))

    plt.tight_layout()
    plt.suptitle(title, x=0.5, y=1.0, fontsize=16)
    plt.show()


def rotate_img(img):
    return jnp.rot90(img, axes=(0, 1))


def identity(img):
    return img


def random_rotate(img, rotate):
    """Randomly rotate an image by 90 degrees.

    Args:
        img: Array representing the image
        rotate: Boolean for rotating or not
    Returns:
        Either Rotated or an identity image
    """
    return jax.lax.cond(rotate, rotate_img, identity, img)


# # Run the pipeline on a single image
# # Get an image
# img = images[0]
# img_copy = img.copy()

# # Pass the image copy to augmentation pipeline
# augmented = random_rotate(img_copy, 1)

# # Plot the original image and the augmented image
# _, ax = plt.subplots(1, 2, figsize=(12, 8))

# ax[0].imshow(img)
# ax[0].axis("off")
# ax[0].set_title("Original Image")

# ax[1].imshow(augmented)
# ax[1].axis("off")
# ax[1].set_title("Augmented Image")

# plt.show()

# Using the same original image
# img = images[0]
# img_copy = img.copy()

# # Batch size of the output as well as for the boolean array
# # used to tell whether to rotate an input image or not
batch_size = 8

# # We use seed for anything that involves `random`
key = random.PRNGKey(1234)

# # Although splitting is not necessary as the key is only used once,
# # I will just leave the original key as it is
key, subkey = random.split(key)
rotate = random.randint(key, shape=[batch_size], minval=0, maxval=2)

# # Return identical or flipped image via augmentation pipeline
# # We will transform the original `random_rotate(..)` function
# # using vmap
# augmented = vmap(random_rotate, in_axes=(None, 0))(img_copy, rotate)

# print("Number of images to generate: ", batch_size)
# print("Rotate-or-not array: ", rotate)
# plot_images(augmented, batch_size=8, title="Multiple augmenetd images from a single input image")

# # Original images
# plot_images(images, batch_size=8, title="Original images")

# # Augment a batch of input images using the same augmentation pipeline
# augmented = vmap(random_rotate, in_axes=(0, 0))(images, rotate)
# plot_images(augmented, batch_size=8, title="Augmented Images")

# # JIT the vmapped function
# vmap_jitted = jit(vmap(random_rotate, in_axes=(0, 0)))

# # Run the pipeline again using the jitted function
# augmented = (vmap_jitted(images, rotate)).block_until_ready()

# # Plot the images and check the results
# plot_images(augmented, batch_size=8, title="Jitting vmapped function")


def rotate_90(img):
    """Rotates an image by 90 degress k times."""
    return jnp.rot90(img, k=1, axes=(0, 1))


def identity(img):
    """Returns an image as it is."""
    return img


def flip_left_right(img):
    """Flips an image left/right direction."""
    return jnp.fliplr(img)


def flip_up_down(img):
    """Flips an image in up/down direction."""
    return jnp.flipud(img)


def random_rotate(img, rotate):
    """Randomly rotate an image by 90 degrees.

    Args:
        img: Array representing the image
        rotate: Boolean for rotating or not
    Returns:
        Rotated or an identity image
    """

    return jax.lax.cond(rotate, rotate_90, identity, img)


def random_horizontal_flip(img, flip):
    """Randomly flip an image vertically.

    Args:
        img: Array representing the image
        flip: Boolean for flipping or not
    Returns:
        Flipped or an identity image
    """

    return jax.lax.cond(flip, flip_left_right, identity, img)


def random_vertical_flip(img, flip):
    """Randomly flip an image vertically.

    Args:
        img: Array representing the image
        flip: Boolean for flipping or not
    Returns:
        Flipped or an identity image
    """

    return jax.lax.cond(flip, flip_up_down, identity, img)


# Get the jitted version of our augmentation functions
random_rotate_jitted = jit(vmap(random_rotate, in_axes=(0, 0)))
random_horizontal_flip_jitted = jit(vmap(random_horizontal_flip, in_axes=(0, 0)))
random_vertical_flip_jitted = jit(vmap(random_vertical_flip, in_axes=(0, 0)))


def augment_images(images, key):
    """Augment a batch of input images.

    Args:
        images: Batch of input images as a jax array
        key: Seed/Key for random functions for generating booleans
    Returns:
        Augmented images with the same shape as the input images
    """

    batch_size = len(images)

    # 1. Rotation
    key, subkey = random.split(key)
    rotate = random.randint(key, shape=[batch_size], minval=0, maxval=2)
    augmented = random_rotate_jitted(images, rotate)

    # 2. Flip horizontally
    key, subkey = random.split(key)
    flip = random.randint(key, shape=[batch_size], minval=0, maxval=2)
    augmented = random_horizontal_flip_jitted(augmented, flip)

    # 3. Flip vertically
    key, subkey = random.split(key)
    flip = random.randint(key, shape=[batch_size], minval=0, maxval=2)
    augmented = random_vertical_flip_jitted(augmented, flip)

    return augmented.block_until_ready()

# Because we are jitting the transformations, we will record the
# time taken for augmentation on subsequent calls


for i in range(3):
    print("Call: ", i + 1, end=" => ")
    key = random.PRNGKey(0)
    start_time = time.time()
    augmented = augment_images(images, key)
    print(f"Time taken to generate augmentations: {time.time()-start_time:.2f}")

# Plot the augmented images
plot_images(augmented, batch_size=8, title="Augmenetd Images")
