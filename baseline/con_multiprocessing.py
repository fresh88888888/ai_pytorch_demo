import time
import multiprocessing


# A CPU heavy calculation, just as n example. This can be anything you like.
def heavy(n, myId):
    for x in range(1, n):
        for y in range(1, n):
            x ** y
    print(myId, "is done")


def multiproc(n):
    process = []

    for i in range(n):
        p = multiprocessing.Process(target=heavy, args=(500, i,))
        process.append(p)
        p.start()

    for p in process:
        p.join()

# A CPU heavy calculation, just as an example. This can be anything you like
def doit(n):
    heavy(500, n)
    

def pooled(n):
    # By default, our pool will have numproc slots
    with multiprocessing.Pool() as pool:
        pool.map(doit, range(n))


if __name__ == '__main__':
    start = time.time()
    #multiproc(80)
    pooled(80)
    end = time.time()
    print('Took:', end - start)
