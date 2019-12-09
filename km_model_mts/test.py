from itertools import combinations
import numpy as np
r1 = list(combinations(np.arange(0, 4, 1), 1))
r2 = list(combinations(np.arange(0, 4, 1), 2))
r3 = list(combinations(np.arange(0, 4, 1), 3))
r4 = list(combinations(np.arange(0, 4, 1), 4))
print(r1)
print(r3)
print(r1+r3)
