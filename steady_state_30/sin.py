import numpy as np
import matplotlib.pyplot as plt
'''
actuation = 1.0
phase = -0.1

plt.plot(np.sin(((np.linspace(0,10,100))/2)+actuation))
plt.show()
'''
new_store_springs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (5, 6), (6, 7), (7, 8), (8, 6), (8, 2), (8, 9)]
point_id = 8
a, b = point_id, 8
print(a,b,(a, b) in new_store_springs)
attempts = 0
while ((a, b) in new_store_springs or (b, a) in new_store_springs or b == a) and attempts < 10:
    print((a, b))
    a = point_id
    b = np.random.randint(0, point_id - 2)
    attempts += 1
    print(a,b)