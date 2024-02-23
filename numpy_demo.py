''' numpy demo '''
import random
import dataclasses
import timeit
import numpy as np
import matplotlib.pyplot as plt

# object oriented approach


@dataclasses.dataclass
class RandomWalker:
    'RandomWalker is class'

    def __init__(self) -> None:
        self.position = 0

    def walk(self, n):
        'define a generator for position.'
        self.position = 0
        for _ in range(n):
            yield self.position
            self.position += random.randint(0, 1) - 1


walker = RandomWalker()
# walk = list(walker.walk(10000))
OPERATION = "[position for position in walker.walk(n=10000)]"
print(timeit.timeit(OPERATION, globals=globals(), number=10))

# procedural approach


def random_walk(n):
    'random walk function'
    position = 0
    walk = [position]
    for _ in range(n):
        position += random.randint(0, 1) - 1
        walk.append(position)
    return walk


# walk = random_walk(10000)
print(timeit.timeit("random_walk(n=10000)", globals=globals(), number=10))

# vectorized approach


def random_walk_faster(n=1000):
    'random walk faster function'
    steps = np.random.choice([-1, 1], n)
    return np.cumsum(steps)


print(timeit.timeit("random_walk_faster(n=10000)", globals=globals(), number=10))

# Define a 1D array
my_array = np.array([1, 2, 3, 4], dtype=np.int64)

# Define a 2D array
my_2d_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)

# Define a 3D array
my_3d_array = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [
                       [1, 2, 3, 4], [9, 10, 11, 12]]], dtype=np.int64)

print(my_array)
print(my_2d_array)
print(my_3d_array)

# Print out memory address
print(my_2d_array.data)

# Print out the shape
print(my_2d_array.shape)

# Print out the data type
print(my_2d_array.dtype)

# Print out the stride
print(my_2d_array.strides)

# Create an array of ones
ones_array = np.ones((3,4))
print('One Array: ', ones_array)

# Create an array with of zeros
zeros_array = np.zeros((2,3,4), dtype=np.int64)
print('Zeros Array: ', zeros_array)

# Create an array with random values
random_array = np.random.random((2,2))
print('Random Array: ', random_array)

# Create an empty array
empty_array = np.empty((3,2))
print('Empty Array: ', empty_array)

# Create a full array
full_array = np.full((2,7), 7)
print("Full Array: ", full_array)

# Create an array of evenly-spaced values
arange_array = np.arange(10, 25, 5)
print('Arange Array: ', arange_array)

# Create an array of evenly-space values
linspace_array = np.linspace(0,2,9)
print('Linspace Array: ', linspace_array)
np.savetxt('linspace.out', linspace_array, delimiter=' ')


# Import your data
x,y,z = np.loadtxt('data.txt',skiprows=1, unpack=True)
print(x,y,z)

# Create a numpy array with shape (2,4) and dtype int64
d_array = np.array([[1, 2,3,4], [5,6,7,8]], dtype=np.int64)

print("Number of dimensions of d_array: ", d_array.ndim)
print("Number of element in d_array: ", d_array.size)
print("Information about the memory layout of d_array: ", d_array.flags)
print("Length of one array element in bytes: ", d_array.itemsize)
print("Total consumed bytes by d_array: ", d_array.nbytes)


my_array = np.array([[1, 2,3,4], [5,6,7,8]], dtype=np.int64)
# Print the length of `my_array`
print('my_array length: ', len(my_array))
# Change the data type of `my_array`
my_array.astype(float)
print(my_array)

# Initialize 3 x 4 array of one and assign it to the variable `a` 
a = np.ones((3, 4))
print(a)
# Initialize a 3 x 4 array of random numbers between 0and 1 and assign it to variable `b`
b = np.random.random((5,1,4))
print(b)
# Add the arrays `x` and `y` element-wise and print the resulting array
c = a + b

print('Result of a + b: ', c)
# Print the shape of resulting array
print("Shape of a + b: ", c.shape)


a = np.array([[1,2,3,4],[5,6,7,8]])
b = np.array([6,7,8,9])

# Add `a` and `b`
c = np.add(a, b)
print('Addition of a and b: ', c)

# Subsract `a` and `b`
c = np.subtract(a, b)
print('Substract of a from b: ', c)

# Multiply `a` and `b`
c = np.multiply(a, b)
print('Element-wise multiplication of a and b: ', c)

# Divide `a` and `b`
c = np.divide(a, b)
print("Divide of a and b: ",c)

# Initialize arrays
a = np.array([1, 1, 0, 0], dtype=bool)
b = np.array([1, 0, 1, 0], dtype=bool)

# `a` AND `b`
print('a and b: ', np.logical_and(a, b))

# `a` OR `b`
print('a or b: ', np.logical_or(a, b))

# `a` NOT `b`
print('a not b: ', np.logical_not(a))
print('a not b: ', np.logical_not(b))

# Print subsets
print(my_array[1])

# Print subsets
print(my_array[1][2])
print(my_array[1, 2])

# Print subsets
print(my_2d_array[0:2, 1])

my_array = np.array([1, 2, 3, 4])
my_3d_array = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [
                       [1, 2, 3, 4], [9, 10, 11, 12]]], dtype=np.int64)

# Try out a simple example
print(my_array[my_array < 2])

# Specify a condition
bigger_than_3 = (my_3d_array >= 3)

# Use the condition to index our 3d array
print(my_3d_array[bigger_than_3])

my_array = np.array([1, 2, 3, 4])
my_2d_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)

my_3d_array = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [
                       [1, 2, 3, 4], [9, 10, 11, 12]]], dtype=np.int64)

# Select elements at (1,0), (0,1), (1,2) and (0,0)
#print(my_2d_array[[1, 0, 1, 0], [0, 1, 2, 0]])

# Select a subset of the rows and columns
print(my_2d_array[[1, 0, 1, 0]][:, [0,1,2,0]])

# Look up info on `mean` with `np.lookfor()`
# print(np.lookfor("mean"))

# Get info on data types with `np.info()`
# np.info(np.ndarray.astype)

my_2d_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
print(my_2d_array.shape)
# Print `my_2d_array`
print(my_2d_array)

# Transpose `my_2d_array`
print(np.transpose(my_2d_array))

# Or use `T` to transpose `my_2d_array`
print(my_2d_array.T)

my_array = np.array([1, 2, 3, 4])
print(my_array.shape)
# Print `my_array
print(my_array)

# Transpose `my_array
print(np.transpose(my_array))

# Or use `T` to transpose `my_array
print(my_array.T)
print(my_array.shape)

x = np.ones((3,4))
# Print the shape of `x`
print(x.shape)

# Resize `x` to ((6, 4))
np.resize(x, (6, 4))

# Try out this as well
x.resize((6, 4))
print(x)

my_array = np.array([1, 2, 3, 4])

# Print `my_array` before appending
print("my_array before appending:", my_array)

# Append a 1D array to `my_array`
new_array = np.append(my_array, [7,8,9,10])

# Print `new_array`
print('new_array: ', new_array)

# Print `my_array` after appending
print("my_array after appending:", my_array)

my_2d_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)

# Print `my_2d_array` before appending
print("my_2d_array before appending:\n", my_2d_array)

# Append an extra column to `my_2d_array`
new_2d_array = np.append(my_2d_array, [[7], [8]], axis=1)
# Print `new_2d_array`
print("new_2d_array:\n", new_2d_array)

my_array = np.array([1, 2,3,4])
# Insert `5` at index 1
print(np.insert(my_array, 1, 5))
# Delete the value at index 1
print(np.delete(my_array, [1]))

x = np.ones((4,))
my_array = np.array([1, 2, 3, 4])
my_resized_array = np.resize(my_array, (2, 4))
my_2d_array = np.array([[1, 2,3,4], [5,6,7,8]], dtype=np.int64)

print(x)
# Concatentate `my_array` and `x`
print(np.concatenate((my_array, x)))
# Stack arrays row-wise
print(np.vstack((my_array, my_2d_array)))

print(my_resized_array)
# Stack arrays row-wise
print(np.r_[my_resized_array, my_2d_array])

# Stack arrays horizontally
print(np.hstack((my_resized_array, my_2d_array)))

# Stack arrays column-wise
print(np.column_stack((my_resized_array, my_2d_array)))

# Stack arrays column-wise
print(np.c_[my_resized_array, my_2d_array])

print('------------------------------')
my_stacked_array = np.hstack((my_resized_array, my_2d_array))
print(my_stacked_array)
# Split `my_stacked_array` horizontally at the 2nd index
print(np.hsplit(my_stacked_array, 2))

# Split `my_stacked_array` vertically at the 2nd index
print(np.vsplit(my_stacked_array, 2))

print('------------------------------')

# # Initialize your array
# my_3d_array = np.array([[[1, 2,3,4], [5,6,7,8]], [[1,2,3,4], [9,10,11,12]]], dtype=np.int64)
# print(my_3d_array)
# # Pass the array to `np.histogram()`
# print(np.histogram(my_3d_array))
# # Specify the number of bins
# print(np.histogram(my_3d_array, bins=range(0, 13)))

# # Construct the histogram with a flattened 3d array and a range of bins
# plt.hist(my_3d_array.ravel(), bins=range(0,13))
# # Add a title to the plot
# plt.title('Frequency of My 3D Array Elements')
# # Show the plot
# plt.show()


# Create an array
points = np.arange(-5, 5, 0.01)

# Make a meshgrid
xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs ** 2 + ys ** 2)
print(z)
# Display the image on the axes
plt.imshow(z, plt.cm.get_cmap('Grays'))
# Draw a color bar
plt.colorbar()
# Show the plot
plt.show()
