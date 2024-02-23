
import time

# A CPU heavy caculation, just as an example. This can be anyting you like.
def heavy(n, myid):
    for x in range(1, n):
        for y in range(1, n):
             x**y
    print(myid, 'is done')
    
def squential(n):
    for i in range(n):
        heavy(500, i)
        
if __name__ == "__main__":
    start = time.time()
    squential(80)
    end = time.time()
    print("Took: ", end - start)
    
