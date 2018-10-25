#!/usr/bin/env python

"""
Demo of how to call low-level MAGMA wrappers to perform SVD decomposition.

Note MAGMA's SVD implementation is a hybrid of CPU/GPU code; the inputs
therefore must be in host memory.
"""

import numpy as np
import skcuda.magma as magma
import time

magma.magma_init()

N=8172
M = np.random.random((N,N))+1j*np.random.random((N,N))
#M = np.array([[3,2,1],[1,2,2],[4,5,0]], dtype=np.complex128)  
M_orig = M.copy()



# Set up output buffers:
w = np.zeros((N,), np.complex128) # eigenvalues
vl = np.zeros((N, N), np.complex128)
vr = np.zeros((N, N), np.complex128)

# Set up workspace:
nb = magma.magma_get_zgeqrf_nb(N,N)
lwork = N*(1 + 2*nb)

work = np.zeros((lwork,), np.complex128)
rwork= np.zeros((2*N,), np.complex128)

# Compute:
gpu_time = time.time();
status = magma.magma_zgeev(b'V', b'V', N, M.ctypes.data, N, w.ctypes.data, vl.ctypes.data, N, vr.ctypes.data, N, work.ctypes.data, lwork, rwork.ctypes.data)
gpu_time = time.time() - gpu_time;
#status = magma.magma_sgesvd('A', 'A', m, n, x.ctypes.data, m, s.ctypes.data,
#                            u.ctypes.data, m, vh.ctypes.data, n,
#                            workspace.ctypes.data, Lwork)

print("GPU time = ", gpu_time)

# Confirm that solution is correct by ensuring that the original matrix can be
# obtained from the decomposition:
print('correct solution: %r' %
      np.allclose(M_orig, np.dot(vl.T, np.dot(np.diag(w), vr)), 1e-4))
magma.magma_finalize()

#print(w)
#print(vr)
#print(vl)
#print( np.dot(vl.T, np.dot(M_orig, vr)))
