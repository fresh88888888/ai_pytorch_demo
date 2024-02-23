#### VSCode: Adding Poetry Python Interpreter

I’ve been trying out Python’s Poetry dependency management tool recently and I really like it, but couldn’t figure out how to get it setup as VSCode’s Python interpreter. In this blog post, we’ll learn how to do that.

One way to add the Python interpreter in VSCode is to press Cmd+Shift+p and then type Python Interpreter. If you select the first result, you’ll see something like the following:

When I create a virtual environment directly it’ll usually appear on the list, but Poetry wasn’t. I went a-searching and came across a StackOverflow thread where people were experiencing the same problem.

I tried some of the suggestions, but they mostly didn’t work for me. I did, however, realise that I’d be able to add the interpreter manually if I knew its path.This led me to the following command which gives us that answer:

```bash
$ poetry env info
```
If we want to only get the path, we can pass in --path:
```bash
$ poetry env info --path
```
We can then pipe that to our clipboard:
```bash
$ poetry env info --path | pbcopy
```
And now we can go back to the Python interpreter window and paste in the result.

poetry excuteing py file:
```bash
$ poetry run python hello.py
```

#### Python Concurrency

Before you consider concurrency in python, which can be quite tricky, always take a good look at your code and algorithms first. Many speed and performance issues are resolved by implementing a better algorithm or adding caching. Entire book are written about this subject, but some general guidelines to follow are:

- **Measure, don't guess** Measure which parts of your code take the most time to run, Focus on those parts first.
- **Implement caching** This can be a big optimization if you perform many repeated lookups from disk, the network, and database.
- **Reuse objects** instead of creating a new one each interation, Python has to clean up every object you created to free memory. This is what we call garbage collection. The garbage collection of many unused object can slow down your software considerably.
- **Reduce the number of iterations** in your code if possible, and reduce the number of operations inside iterations.
- **Avoid(deep) recursion** It requires a lot of memory and housekeeping for the python interpreter, Use things like generators and iteration instead.
- **Reduce memory usage** In general, Try to reduce the usage of memory, For example: parse huge file line by line instead of loading it in memory first.
- Do you really need to perform that operation? Can it be down later? Or can it be down once, and can the result of it be stored instead of calcuated over and over again.
- **Use PyPy or Cpython** You can also consider an alternative Python implementation. There are speedy Python variants out there. See below for more info on this.

$e^{\pi i} + 1 = 0$

Writing one line,
then another without a break,
results in text flowing as a single line.