import threading
import time


# A CPU heavy calculation, just as an example. This can be anything you like.
def heavy_cpu(n, myId):
    for x in range(1, n):
        for y in range(1, n):
            x ** y
    print(myId, "is done")


# An I/O intensive calculation. We simulate it with sleep.
def heavy_io(n, myId):
    time.sleep(2)
    print(myId, "is done")


def threaded(n):
    threads = []

    for i in range(n):
        t = threading.Thread(target=heavy_io, args=(500, i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    start = time.time()
    threaded(80)
    end = time.time()
    print("Took: ", end - start)
