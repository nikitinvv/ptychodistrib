import numpy as np
import cupy as cp
import dxchange
from random import sample
import matplotlib.pyplot as plt
import ptychodistrib

if __name__ == "__main__":

    n = 384  # object size n x
    nz = 384  # object size in z
    ndet = 128  # detector size
    nprb = 128  # probe size
    nscan = 1024  #  number of scan positions (max 4554)
    nnodes = 4  # number of nodes (multiple of nscan)

    # Load object
    amp = dxchange.read_tiff('data/object_ampe.tiff')
    angle = dxchange.read_tiff('data/object_anglee.tiff')
    psiinit = amp*np.exp(1j*angle)

    # Load probe
    probe_amp = dxchange.read_tiff('data/probe_amp.tiff')
    probe_angle = dxchange.read_tiff('data/probe_angle.tiff')
    prb = probe_amp*np.exp(1j*probe_angle)

    # Load scan positions
    scan = np.load('data/scan.npy')
    # pick randomly nscan positions
    scan = scan[:,sample(range(scan.shape[1]),nscan)]
    plt.plot(scan[1], scan[0], 'r.')
    plt.savefig(f'data/scan.png')

    # copy to gpu
    psiinit = cp.array(psiinit)
    prb = cp.array(prb)
    scan = cp.array(scan)        

    # compute data
    with ptychodistrib.SolverPtycho(nz, n, nscan, ndet, nprb, 1) as pslv:
        # data = ||FQpsi||^2
        data = cp.abs(pslv.fwd_ptycho(psiinit, prb, scan))**2

    # ADMM solver
    with ptychodistrib.SolverPtycho(nz, n, nscan, ndet, nprb, nnodes) as pslv:
        # init variable
        psi = cp.ones([nnodes, *psiinit.shape], dtype='complex64')
        z = cp.ones(psiinit.shape, dtype='complex64')
        lamd = cp.zeros([nnodes, *psiinit.shape], dtype='complex64')

        niter = 128  # number of outer iterations
        piter = 4  # number of inner iterations in ptychography

        rho = 0.5
        for m in range(niter):
            # keep z from the previous iteration for penalty updates
            z0 = z.copy()
            # 1) ptycho problem (many nodes)
            psi = pslv.grad_ptycho_batch(
                data, psi, prb, scan, z-lamd, rho, piter)
            # 2) regularization problem (one node)
            z = cp.mean(psi+lamd, axis=0)
            # 3) lambda update
            lamd = lamd + (psi - z)
            # update rho, tau for a faster convergence
            rho = pslv.update_penalty(psi, z, z0, rho)
            # Lagrangians difference between two iterations
            lagr = pslv.take_lagr(data, psi, prb, scan, z, lamd, rho)
            print("%d/%d) rho=%.2e, Lagrangian terms:  %.2e %.2e %.2e, Sum: %.2e" %
                  (m, niter, rho, *lagr))

    dxchange.write_tiff(cp.angle(z).get(),
                        'rec_admm/object_angle.tiff', overwrite=True)
    dxchange.write_tiff(cp.abs(z).get(),
                        'rec_admm/object_amp.tiff', overwrite=True)
