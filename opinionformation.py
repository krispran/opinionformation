import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Opinions of agents
O = np.matrix([ 1,0.31,0.62]).T # [o1, o2, o3]

# Confidence (trust) between agents
A = np.matrix([ [1  ,0  ,0 ],   # [a11, a12, a13]
                [0.2,0.5,0.3],  # [a21, a21, a31]
                [0.3,0.4,0.4]]) # [a31, a32, a33]

# Number of iterations of the model
nIter = 10

# Iteration variable
x = []

# Opinion values of agents variables
seller = []
buyer1 = []
buyer2 = []

# For loop of opinion exchange 
for i in range(nIter):

    # Appendance of iterations and opinions
    x.append(i)
    seller.append(O[0,0])
    buyer1.append(O[1,0])
    buyer2.append(O[2,0])

    # Economic utility associated to confidence of buyer1 if opinions differ
    if 1 - abs(O[0,0] - O[1,0]) > 0.01:
        U = np.matrix([O[0,0] / O[1,0], 0, 0]).T
        A[1,0] = (A[1,:] @ U)

    # Economic utility associated to confidence of buyer2
    if 1 - abs(O[0,0] - O[2,0]) > 0.01:
        U = np.matrix([O[0,0] / O[2,0], 0, 0]).T
        A[2,0] = (A[2,:] @ U)

    # Normalisation of updated confidence between agents
    A = normalize(A, norm="l1")

    # Opinion exchange
    O = A @ O

# Plotting the variables for a time series model
plt.figure(figsize=(8,4))
plt.plot(x, seller, label="seller")
plt.plot(x, buyer1, label="buyer 1")
plt.plot(x, buyer2, label="buyer 2")
plt.xlabel("iteration")
plt.ylabel("opinion")
plt.legend()
plt.show()
