import datetime
import logging
import time

start_time = time.time()
print("Start training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))

for i in range(1000000):
    for i in range(100):
        x = (i * 1)
        
end_time = time.time()
print("End training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
print("Training took {0:.1f}".format(end_time-start_time))