""" root module for project"""
import os
import shutil
import json
import traceback
import numpy
import jmespath
import yaml
import csv

# Sometimes programing might feel a bit like doing migic, especially when you're just starting out,
# But once you take a peek under the hood and see how thing actually work, A lot of magicis gone,
# Let's continue and find out what objects really are, and how they are defined.

# Object : an object is a collection of data(variable) nd methods that opreate on the data. Objects are definedby a Python class.
# Class : A class is the blueprint for one or more objects.

# Creating a Python class


class Vehicle:
    """ parent class
    """
    speed = 0
    started = False

    def __init__(self, started=False, speed=0):
        self.started = started
        self.speed = speed

    def start(self):
        self.started = True
        print("Started, let's ride!")

    def stop(self):
        self.speed = 0

    def increase_speed(self, delta):
        if self.started:
            self.speed = self.speed + delta
            print("Vrooooom!")
        else:
            print("You need to start me first")


class Car(Vehicle):
    """Car class

    Args:
        Vehicle (class): parent class for Car
    """
    trunk_open = False

    def open_trunk(self):
        """ open tunk method for ths Car class 
        """
        self.trunk_open = True

    def close_trunk(self):
        self.trunk_open = False


class Motorcycle(Vehicle):
    """ motorcycle class inheritance Vehicle class

    Args:
        Vehicle (class): parent class
    """
    center_stand_out = False

    def __init__(self, center_stand_out=False):
        self.center_stand_out = center_stand_out
        super().__init__()

    def start(self):
        print('Sory, Out of rue!')


# Python Constructor for __init__(...) and Python Inheritance
car1 = Car()
car2 = Car(True)
car3 = Car(True, 59)
car4 = Car(started=False, speed=23)

# Python modules: Bundle code and import it from other files
# 1.How to create modules and organize your code
# 2.The python import statement that allows up to import(parts of) a module
# 3.Some best practices when it comes to modules
# 4.How to create runnable modules

# my_function()
# if os.path.isfile('mymodule.py'):
#     print('It is a file')

# if os.path.isdir('.vscode'):
#     print('It is a dir')
    
# if os.path.isdir('mydir'):
#     print('mydir is existed')
#     os.rename('mydir', 'yourdir')
# else :
#     os.mkdir('mydir')

# with open(file='mymodule.py', encoding='utf8', mode='a') as f:
#     f.write('# write some words in mymodule.py \n')
   
# with open(file='mymodule.py', encoding='utf8', mode='r') as f:
#     for line in f:
#         print(line)
    
# shutil.move('mymodule.py', 'yourdir/mymodule.py')

# shutil.copy('mymodule.py', 'mymodule_copy.py')

# shutil.copytree('yourdir', 'yourdir_copy')

# shutil.rmtree('yourdir')

try:
    print(2/0)
except ZeroDivisionError:
    print('You can\'t divide by zero.')


try:
    
    with open(file='mymodule.py', encoding='utf8', mode='r') as f:
            for line in f:
                print(line)
        
except IOError as e:
    print('An error occured: ', e)

finally:
    print('finally excuting.')


user_json = '{"name": "John", "age": 39}'
user = json.loads(user_json)
try:
    print(user['name'])
    print(user['age'])
    print(user['address'])
except KeyError as e:
    print('There are missing fields in the user object: ', e)
    

class UserNotFoundError(Exception):
    'the class good'
    pass

def fetch_user(user_id):
    # Here you would fetch from some kind of db, e.g.:
    # user = db.get_user(user_id)
    
    # To make this example runnable, let's set it to None
    user = None
    if user == None:
        raise UserNotFoundError(f'User {user_id} not in database')
    else:
        return user
    

user = [123, 456, 789]
for user_id in user:
    try:
        fetch_user(user_id)
    except UserNotFoundError as e:
        print('There was an error: ', e)
        # traceback.print_exc()


def print_arguments(func):
    'what you name'
    def wrapper(the_number):
        print('Argument for', func.__name__, "is", the_number)
        return func(the_number)
    return wrapper

@print_arguments
def add_one(x):
    'how are you'
    return x + 1

print(add_one(2))

print('--------------------------')

class EvenNumbers:
    last = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.last += 2
        if self.last > 8:
            raise StopIteration
        return self.last
    
even_number = EvenNumbers()
for num in even_number:
    print(num)


class MyDocumentedClass:
    """This class is well documented but doesn't do anything special."""

    def do_nothing(self):
        """This method doesn't do anything but feel free to call it anyway."""
        pass
    
# help(MyDocumentedClass)

persons = {
   "persons": [
     { "name": "erik", "age": 38 },
     { "name": "john", "age": 45 },
     { "name": "rob", "age": 14 }
   ]
}

print(jmespath.search('persons[*].age', persons))
print(jmespath.search("persons[?name=='john'].age", persons))

names_yaml = """
 - 'eric'
 - 'justin'
 - 'mary-kate'
"""

names = yaml.safe_load(names_yaml)
print(names)
with open('names.yaml', 'w') as file:
    yaml.dump(names, file)

with open('names.yaml', 'r') as file:
    print(file.read())

# data={}
# with open('config.yaml', 'r') as file:
#     docs = yaml.safe_load_all(file)
#     i = 0
#     for doc in docs:
#         data[i] = doc
#         i+=1
    
# with open('open.json', 'a') as json_file:
#         json.dump(data, json_file)


# with open('open.json', 'r') as json_file:
#     configuration = json.load(json_file)
    
# with open('config.yaml', 'w') as yaml_file:
#     yaml.dump(configuration, yaml_file)

# with open('config.yaml', 'r') as yaml_file:
#     print(yaml_file.read())

# # Define the customer dialect
# my_dialect = csv.register_dialect(
#     'my_dialect', delimiter='&', quotechar='"', quoting=csv.QUOTE_MINIMAL)

# with open('person.csv', newline='') as csv_file:
#     csvfile = csv.reader(csv_file, dialect='my_dialect')
    
#     for row in csvfile:
#         print(row)


# with open('person.csv', 'a', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file, dialect='my_dialect')
    
#     csv_writer.writerow(["Name", "Age", "Country"])
#     csv_writer.writerow(["John Doe", 30, "United States"])
#     csv_writer.writerow(["Jane Doe", 28, "Canada"])


# print(f"""{f'''{f'{f"{1+1}"}'}'''}""")
# url = 'sdsdsd'
# raise ValueError("bad value for url: %s" % url)

planets = ['Mercury', 'Venus', 'Earth', 'Mars','Jupiter', 'Saturn', 'Uranus', 'Neptune']

# str.upper() returns an all-caps version of a string
loud_short_plants = [planet.upper() + '!' for planet in planets if len(planet) < 6]
t = 1, 2, 3
print(t)