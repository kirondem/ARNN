import random
import numpy as np
import time

def random1(x, H_t, random_idxs):
    x_idxs = np.where(x > H_t[random_idxs])[0]
    np.put(H_t, random_idxs[x_idxs], x[x_idxs])
    return H_t, random_idxs[x_idxs]

def random2(x, H_t, random_idxs):
    x_idxs = np.where(x > H_t[random_idxs])[0]
    for x_idx in x_idxs:
        H_t[random_idxs[x_idx]] = x[x_idx]
    return H_t, random_idxs[x_idxs]
    #print(H_t)

def random3(x, H_t, random_idxs):
    activated_units  = []
    for i, random_idx in enumerate(random_idxs):  
        if (x[i] > H_t[random_idx]):
            H_t[random_idx] = x[i]
            activated_units.append(random_idx)
    return H_t, activated_units
    #print(H_t)

if __name__ == '__main__':

    H_t = np.zeros(1000000)
    x = np.random.rand( 10000 )
    random_idxs = random.sample(range(len(H_t)), len(x))
    random_idxs = np.array(random_idxs)

    t = time.process_time()
    H_t = np.zeros(1000000)
    H_t1, activated_units_idxs = random1(x, H_t, random_idxs)
    #print(H_t)
    #print(activated_units_idxs)
    print('Time: ', time.process_time() - t)

    t = time.process_time()
    H_t = np.zeros(1000000)
    H_t2, activated_units_idxs = random2(x, H_t, random_idxs)
    #print(activated_units_idxs)
    print('Time: ', time.process_time() - t)
    
    t = time.process_time()
    H_t = np.zeros(1000000)
    H_t3, activated_units_idxs =  random3(x, H_t, random_idxs)
    #print(activated_units_idxs)
    print('Time: ', time.process_time() - t)





