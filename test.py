import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.magma as magma

magma.magma_init()  
n = 100              
a = np.asarray(np.random.rand(n, n), order='F', )
a_gpu = gpuarray.to_gpu(a)
ipiv_gpu = gpuarray.empty((n,), np.int32, order='F')
magma.magma_dgetrf_gpu(n ,n, a_gpu.gpudata, n, ipiv_gpu.gpudata)
magma.magma_finalize()
