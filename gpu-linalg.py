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

def eig(a, left=False, right=True, check_finite=True, verbose=True, *args, **kwargs):
    """
        Eigenvalue solver using Magma GPU library (variants of Lapack's geev).
    """
    if useScipy:
        return scipy.linalg.eig(a, left=left, right=right, check_finite=check_finite)
    else:
        if len(a.shape) != 2:
            raise ValueError("M needs to be a rank 2 square array for eig.")

        magma.magma_init()


        dtype = type(a[0,0])

        t = typedict_[dtype]
        N = a.shape[0]

        # Set up output buffers:
        if t in ['s', 'd']:
            wr = np.zeros((N,), dtype) # eigenvalues
            wi = np.zeros((N,), dtype) # eigenvalues
        elif t in ['c', 'z']:
            w = np.zeros((N,), dtype) # eigenvalues
            
        if left:
            vl = np.zeros((N, N), dtype)
            jobvl = 'V'
        else:
            vl = np.zeros((1, 1), dtype)
            jobvl = 'N'
        if right:
            vr = np.zeros((N, N), dtype)
            jobvr = 'V'
        else:
            vr = np.zeros((1, 1), dtype)
            jobvr = 'N'

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
            status = magma.magma_sgeev(jobvl, jobvr, N, a.ctypes.data, N, 
                                    wr.ctypes.data, wi.ctypes.data, 
                                    vl.ctypes.data, N, vr.ctypes.data, N, 
                                    work.ctypes.data, lwork)
        if t == 'd':
            status = magma.magma_dgeev(jobvl, jobvr, N, a.ctypes.data, N, 
                                    wr.ctypes.data, wi.ctypes.data, 
                                    vl.ctypes.data, N, vr.ctypes.data, N, 
                                    work.ctypes.data, lwork)
        if t == 'c':
            status = magma.magma_cgeev(jobvl, jobvr, N, a.ctypes.data, N, 
                                    w.ctypes.data, vl.ctypes.data, N, vr.ctypes.data, N, 
                                    work.ctypes.data, lwork, rwork.ctypes.data)
        if t == 'z':
            status = magma.magma_zgeev(jobvl, jobvr, N, a.ctypes.data, N, 
                                    w.ctypes.data, vl.ctypes.data, N, vr.ctypes.data, N, 
                                    work.ctypes.data, lwork, rwork.ctypes.data)
        gpu_time = time.time() - gpu_time;
        
        if verbose:
            print("Time for eig: ", gpu_time)
        
        if t in ['s', 'd']:
            w_gpu = wr + 1j*wi
        else:
            w_gpu = w

        magma.magma_finalize()

        return w_gpu, vr



if __name__=='__main__':
    print("# Using Magma library: ", (not useScipy))
    N = 100
    #M_gpu = np.random.random((N,N))+1j*np.random.random((N,N))
    M_gpu = np.random.random((N, N))
    W, V = eig(M_gpu)
    