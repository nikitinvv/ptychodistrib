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
    amp = dxchange.read_tiff('data/object_ampe.tiff')
    angle = dxchange.read_tiff('data/object_anglee.tiff')
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

    with ptychodistrib.SolverPtycho(nz, n, nscan, ndet, nprb, 1) as pslv:
        # data = ||FQpsi||^2
        data = cp.abs(pslv.fwd_ptycho(psi, prb, scan))**2
        # gradient solver
        psi = cp.ones_like(psi)
        niter = 32
        psi = pslv.grad_ptycho(data, psi, prb, scan, None, -1, niter)        
    
    dxchange.write_tiff(cp.angle(psi).get(), 'rec/object_angle.tiff',overwrite=True)
    dxchange.write_tiff(cp.abs(psi).get(), 'rec/object_amp.tiff',overwrite=True)    
        