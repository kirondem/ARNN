
 
import threading, queue

def calc_square(num, out_queue1, time=0):
  l = []
  sleep(time)
  for x in num:
    l.append(x*x)
  out_queue1.put(l)
  print("All done in the new thread:", threading.current_thread().name)


arr = [1,2,3,4,5,6,7,8,9,10]
out_queue1=queue.Queue()
t1=threading.Thread(target=calc_square, args=(arr, out_queue1, 5))
t1.start()


arr = [1,2,3]
out_queue2=queue.Queue()
t2=threading.Thread(target=calc_square, args=(arr, out_queue2, 2))
t2.start()

t1.join()
t2.join()


print (out_queue1.get())
print (out_queue2.get())