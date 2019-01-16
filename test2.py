import numpy as np
import scipy.linalg as LA
import sys

from gpu import eig

if __name__=='__main__':
    N = int(sys.argv[1])
    #M_gpu = np.random.random((N,N))+1j*np.random.random((N,N))

    np.random.seed(1234)

    M_gpu = np.zeros((N,N))
    #M_gpu += np.random.random((N, N))
    M_gpu += np.array([[1,2,3],[5,4,2],[1,0,9]])
    

    np.random.seed(1234)
    M_cpu = np.array([[1,2,3],[5,4,2],[1,0,9]]) # np.random.random((N, N))

    print(M_cpu == M_gpu)
    print(M_cpu.dtype)
    
    W_gpu, V_gpu = eig(M_gpu)
    W_cpu, V_cpu = LA.eig(M_cpu, left=False, right=True)

    ind_gpu = np.argsort(W_gpu)
    ind_cpu = np.argsort(W_cpu)

    W_gpu = W_gpu[ind_gpu]
    V_gpu = V_gpu[:, ind_gpu]
#    V_gpu /= V_gpu[0,:]

    W_cpu = W_cpu[ind_cpu]
    V_cpu = V_cpu[:, ind_cpu]
#    V_cpu /= V_cpu[0,:]

    print("CPU:")
    print(W_cpu)
    print(V_cpu)
    print("GPU:")
    print(W_gpu)
    print(V_gpu)
    print("V_gpu - V_cpu = ")
    print(V_gpu- V_cpu)