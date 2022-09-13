import numpy as np
import os.path

n = 11
fitness=[]
for i in range(n):
    filename = "fitS"+str(i)+".npy"
    if os.path.isfile(filename):
        f  = np.load(filename)
        fitness.append(np.amax(f))

print(fitness)
print(np.mean(fitness))
