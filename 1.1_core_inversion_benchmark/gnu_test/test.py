import numpy as np
import sys

nmom = int(sys.argv[1])
q = "hankel-determinant"
#func = lambda x, nmom: x**(2/nmom)
func = lambda x, nmom: x

data = np.genfromtxt("data_nmom{0:d}.out".format(nmom), skip_header=8)
data = data[data[:,0] < 2]
data1 = np.genfromtxt("../../data/{0:s}_nmom{1:d}.dat".format(q, nmom), comments='#')

print(np.max(data[:,-1]))
print(np.corrcoef(data[:,-1], func(data1, nmom)))
