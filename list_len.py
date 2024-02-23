
# Define a simple list to be used throughout the tutorial
import timeit
import numpy as np
from functools import reduce
my_list = ["I", "Love", "Learning", "Python"] * 100

# Use len() to find the size of the list
length = len(my_list)
print(length)

# Naive method using a for loop to count the list's size
couter = 0
for item in my_list:
    couter += 1

print(couter)

# Using list comprehension to count the list's size
length = sum([1 for item in my_list], 0)
print(length)


# Import the reduce function from the functiontools module
# Define a simple function to use with reduce

def update_count(count_so_far, _):
    """Increases the count by 1. The second parameter is not used."""
    return count_so_far + 1


# Use reduce to count the items in the list, We start counting from 0, which is why we have '0'at the end
list_lngth = reduce(update_count, my_list, 0)
print(list_lngth)


# Step 1: Turn the list into an iterator
list_iterator = iter(my_list)

# Initialize a counter to keep track of the number of items.
count = 0

# Step2: Loop through the list using the iterator
while True:
    try:
        # Use next() to get the next item from the iterator.
        next(list_iterator)
        # If next() was successful, Increase the count.
        count += 1
    except StopIteration:
        # Step3: If we reach the end of the list, break out of loop.
        break
print(count)

# Step 1: Enumerate the list and convert it to a list of tuples (index, element)
enumerated_list = list(enumerate(my_list))

# Step 2: Extract the last tuple (which contains the last index and the last element)
list_tuple = enumerated_list[-1]

# Step 3: The size of the list is the last index plus 1 (because of zero-based indexing)
list_size = list_tuple[0] + 1
print(list_size)

# Import the Numpy library

# Step 1: Convert the list into a Numpy array
my_array = np.array(my_list)

# Step 2: Use the 'size' attribute of the Numpy array to find its size
array_size = my_array.size
print(array_size)

# Using map() to count the list's size
length = sum(map(lambda x: 1, my_list))
print(length)


# Method 1: Using len()
def method_len():
    return len(my_list)

# Method 2: Looping through the list


def method_loop():
    counter = 0
    for _ in my_list:
        counter += 1
    return counter

# Method 3: Using a list comprehension


def method_list_comprehension():
    return sum([1 for _ in my_list])

# Method 4: Using reduce()


def method_reduce():
    return reduce(lambda acc, _: acc + 1, my_list, 0)

# Method 5: Using iter() and next()


def method_iter_next():
    iterator = iter(my_list)
    counter = 0
    while True:
        try:
            next(iterator)
            counter += 1
        except StopIteration:
            break
    return counter

# Method 6: Using enumerate()


def method_enumerate():
    return max(enumerate(my_list, 1))[0]

# Method 7: Using numpy


def method_numpy():
    np_array = np.array(my_list)
    return np_array.size

# Method 8: Using map() and sum()


def method_map_sum():
    return sum(map(lambda _: 1, my_list))


# List to hold method names and their excution times.
timing_results = []
methods = [method_len, method_loop, method_list_comprehension, method_reduce,
           method_iter_next, method_enumerate, method_numpy, method_map_sum]
 
# Time each method
for method in methods:
    # Execute the operation 100000 times for better statistical significance
    time_taken = timeit.timeit(method, number=100000)
    timing_results.append((method.__name__, time_taken))
    
# Sort results by time taken for better readability 
timing_results.sort(key=lambda x: x[1])

for method_name, time_taken in timing_results:
    print(f'{method_name}: {time_taken:.5f} seconds')

