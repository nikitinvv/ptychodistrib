import numpy as np
import cupy as cp
import dxchange
import ptychodistrib
import matplotlib.pyplot as plt

if __name__ == "__main__":

    n = 384  # object size n x
    nz = 384  # object size in z
    ndet = 128 # detector size
    nprb = 128 # probe size
    nscan = 4554 # number of scan positions

    # Load object
    amp = dxchange.read_tiff('data/object_amp.tiff')
    angle = dxchange.read_tiff('data/object_angle.tiff')
    psi = amp*np.exp(1j*angle)
    
    # Load probe
    probe_amp = dxchange.read_tiff('data/probe_amp.tiff')
    probe_angle = dxchange.read_tiff('data/probe_angle.tiff')
    prb = probe_amp*np.exp(1j*probe_angle)

    # Load scan positions
    scan = np.load('data/scan.npy')

    plt.plot(scan[1], scan[0], 'r.')
    plt.savefig(f'data/scan.png')    

    # copy to gpu 
    psi = cp.array(psi)
    prb = cp.array(prb)
    scan = cp.array(scan)        

    with ptychodistrib.SolverPtycho(nz, n, nscan, ndet, nprb) as pslv:
        # FQpsi
        data = pslv.fwd_ptycho(psi, prb, scan)
        # Q*F*data
        psia = pslv.adj_ptycho(data, prb, scan)

    #Adjoint test                
    s1 = cp.sum(psi*cp.conj(psia)).get()
    s2 = cp.sum(data*cp.conj(data)).get()
    
    print(
        f'Adjoint tests: <psi,Q*F*FQpsi>=<FQpsi,FQpsi>: {s1:e} ? {s2:e}')
