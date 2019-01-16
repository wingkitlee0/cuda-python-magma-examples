import numpy as np
import scipy.linalg as LA
import sys

from gpu import eig


N = int(sys.argv[1])
#M_gpu = np.random.random((N,N))+1j*np.random.random((N,N))

np.random.seed(1234)


M_gpu = np.zeros((N,N), dtype=np.complex128, order='F')
M_gpu += np.random.random((N, N))


np.random.seed(1234)
M_cpu = np.zeros((N,N), dtype=np.complex128)
M_cpu += np.random.random((N, N))

print(M_cpu == M_gpu)

W_gpu, V_gpu = eig(M_gpu, left=False, right=True, type=np.complex128)
W_cpu, V_cpu = LA.eig(M_cpu, left=False, right=True)


ind_gpu = np.argsort(W_gpu.real)
ind_cpu = np.argsort(W_cpu.real)

W_gpu = W_gpu[ind_gpu]
V_gpu = V_gpu[ind_gpu, :]

W_cpu = W_cpu[ind_cpu]
V_cpu = V_cpu[:, ind_cpu]

print("CPU:")
print(W_cpu)
print(V_cpu)

for i in range(N):
    print(np.sum(V_cpu[:,i]**2))

print("GPU:")
print(W_gpu)
print(V_gpu)

for i in range(N):
    amp = np.sqrt(np.sum(V_gpu[i,:]**2))
    print(amp)
#    V_gpu[:,i] /= amp

#for i in range(N):
#    amp = np.sqrt(np.sum(V_gpu[:,i]**2))
#    print(amp)
    
print("---------")
print("V_gpu - V_cpu = ")
print(V_gpu.T - V_cpu)