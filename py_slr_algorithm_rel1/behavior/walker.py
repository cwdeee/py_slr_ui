import numpy as np

def random_walk_with_boundary(boundary = 190, lc_difficutly = 0.5):
    position = 0
    steps = 0
    while abs(position) < boundary:
        step = np.random.choice([-1* (lc_difficutly),1])#(-1+lc_difficutly)
        position += step
        steps += 1
        #print("s:"+str(step))
        #print("p:"+str(position))
        #print("sts:"+str(steps))
    return steps
