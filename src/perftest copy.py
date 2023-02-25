from time import sleep
from threading import Thread
 
# target function
def task(id, time):
    # block for a moment
    sleep(time)
    # report a message
    print('All done in the new thread:', id)
 
# create a new thread
thread1 = Thread(target=task, args=(1, 6))
# start the new thread
thread1.start()

thread2 = Thread(target=task, args=(2, 1))
# start the new thread
thread2.start()


# wait for the new thread to finish
print('Main: Waiting for thread to terminate...')

thread1.join()
thread2.join()

# continue on
print('Main: Continuing on')