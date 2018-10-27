import numpy as np
import time

try:
    import skcuda.magma as magma
    useScipy = False
except Exception as err:
    print("#", err)
    print("# Cannot import scikit-cuda. Fall back to scipy.linalg")
    import scipy.linalg
    useScipy = True

typedict = {'s': np.float32, 'd': np.float64, 'c': np.complex64, 'z': np.complex128}
typedict_= {v: k for k, v in typedict.items()}

def eig(M, verbose=True, *args, **kwargs):
    """
    """
    if useScipy:
        return scipy.linalg.eig(M, *args, **kwargs)
    else:
        if len(M.shape) != 2:
            raise ValueError("M needs to be a rank 2 square array for eig.")
        dtype = M.dtype

        t = typedict_[dtype]
        N = M.shape[0]

        # Set up output buffers:
        if t in ['s', 'd']:
            wr = np.zeros((N,), dtype) # eigenvalues
            wi = np.zeros((N,), dtype) # eigenvalues
        elif t in ['c', 'z']:
            w = np.zeros((N,), dtype) # eigenvalues
            
        vl = np.zeros((N, N), dtype)
        vr = np.zeros((N, N), dtype)

        # Set up workspace:
        if t == 's':
            nb = magma.magma_get_sgeqrf_nb(N,N)
        if t == 'd':
            nb = magma.magma_get_dgeqrf_nb(N,N)
        if t == 'c':
            nb = magma.magma_get_cgeqrf_nb(N,N)
        if t == 'z':
            nb = magma.magma_get_zgeqrf_nb(N,N)
        
        lwork = N*(1 + 2*nb)
        work = np.zeros((lwork,), dtype)
        if t in ['c', 'z']:
            rwork= np.zeros((2*N,), dtype)

        # Compute:
        gpu_time = time.time();
        if t == 's':
            status = magma.magma_sgeev('N', 'V', N, M_gpu.ctypes.data, N, 
                                    wr.ctypes.data, wi.ctypes.data, 
                                    vl.ctypes.data, N, vr.ctypes.data, N, 
                                    work.ctypes.data, lwork)
        if t == 'd':
            status = magma.magma_dgeev('N', 'V', N, M_gpu.ctypes.data, N, 
                                    wr.ctypes.data, wi.ctypes.data, 
                                    vl.ctypes.data, N, vr.ctypes.data, N, 
                                    work.ctypes.data, lwork)
        if t == 'c':
            status = magma.magma_cgeev('N', 'V', N, M_gpu.ctypes.data, N, 
                                    w.ctypes.data, vl.ctypes.data, N, vr.ctypes.data, N, 
                                    work.ctypes.data, lwork, rwork.ctypes.data)
        if t == 'z':
            status = magma.magma_zgeev('N', 'V', N, M_gpu.ctypes.data, N, 
                                    w.ctypes.data, vl.ctypes.data, N, vr.ctypes.data, N, 
                                    work.ctypes.data, lwork, rwork.ctypes.data)
        gpu_time = time.time() - gpu_time;
        
        if verbose:
            print("Time for eig: ", gpu_time)
        
        if t in ['s', 'd']:
            w_gpu = wr + 1j*wi
        else:
            w_gpu = w

        return w_gpu, vr



if __name__=='__main__':
    print("# Using Magma library: ", (not useScipy))
    N = 100
    #M_gpu = np.random.random((N,N))+1j*np.random.random((N,N))
    M_gpu = np.random.random((N, N))
    W, V = eig(M_gpu)
    