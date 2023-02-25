import datetime
import logging
import time

import cProfile

start_time = time.time()
print("Start training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))

def foo():
    bar()

def bar():
    for i in range(1000000):
        pass
    
end_time = time.time()
print("End training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
print("Training took {0:.1f}".format(end_time-start_time))


cProfile.run('foo()')