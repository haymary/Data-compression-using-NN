import numpy as np
path_to_result = "results/test/"

ar = np.load(path_to_result + "encoded.npy")
print len(ar)
print ar.shape
print ar[1].shape
print ar[1][0]

a = np.reshape(ar, (102400, 25))
print a.shape
print a[1]

at = np.reshape(a, (102400, 1, 25))
print at[1]
