import numpy as np
import matplotlib.pyplot as plt
import sys

# Input filename is read from command line parameters
try:
    input_file = sys.argv[1]
except IndexError:
    msg = "Input file must be provided as command line parameter."
    raise RuntimeError(msg)

# Read input data
with open(input_file, 'r') as fi:
    col_names = fi.readline().split()
data = np.genfromtxt(input_file)
matrix_size = np.round(data[:,0]).astype(int)
cpu_times = data[:,1:]

# Compute average CPU times for each matrix size
n_all = np.unique(matrix_size)
cpu_times_avg = np.empty((len(n_all), cpu_times.shape[1]))
for i,n in enumerate(n_all):
    idx = np.argwhere(matrix_size==n)
    cpu_times_avg[i] = np.mean(cpu_times[idx], axis=0)


# Plot data
solver = col_names[1:]
fig = plt.figure()
ax = fig.add_subplot(111)
for col in range(cpu_times_avg.shape[1]):
    ax.semilogy(n_all, cpu_times_avg[:,col], label=solver[col])
ax.legend()
ax.grid(ls=':', lw=0.5, which='both')

plt.show()
