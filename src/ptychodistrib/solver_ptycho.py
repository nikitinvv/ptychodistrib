import cupy as cp
import numpy as np
from .ptychofft import ptychofft


class SolverPtycho(ptychofft):
    """Ptychography solver class.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attributes
    ----------
    nz, n : int
        The pixel height and width of the projection.
    nscan : int
        Number of scanning positions
    ndet : int
        Detector size
    nprb : int
        Probe size    
    nnodes : int
        Number of nodes
    """

    def __init__(self, nz, n, nscan, ndet, nprb, nnodes):
        """Please see help(SolverPtycho) for more info."""
        self.nnodes = nnodes
        if(nscan % self.nnodes != 0):
            print(f'Number of nodes should be a multiple of nscan')
            exit()
        super().__init__(1, nz, n, nscan//nnodes, ndet, nprb, 1)  # ntheta==1, ngpu==1

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_ptycho(self, psi, prb, scan):
        """Ptychography transform (FQ)"""
        res = cp.zeros([self.nscan, self.ndet, self.ndet], dtype='complex64')
        # convert to C-contiguous arrays if needed
        psi = cp.ascontiguousarray(psi)
        prb = cp.ascontiguousarray(prb)
        scan = cp.ascontiguousarray(scan)
        # run C wrapper
        self.fwd(res.data.ptr, psi.data.ptr,
                 prb.data.ptr, scan.data.ptr, 0)  # igpu = 0
        return res

    def adj_ptycho(self, data, prb, scan):
        """Adjoint ptychography transform (Q*F*)"""
        res = cp.zeros([self.nz, self.n], dtype='complex64')
        # convert to C-contiguous arrays if needed
        data = cp.ascontiguousarray(data)
        prb = cp.ascontiguousarray(prb)
        scan = cp.ascontiguousarray(scan)
        # run C wrapper
        self.adj(res.data.ptr, data.data.ptr,
                 prb.data.ptr, scan.data.ptr, 0)  # igpu = 0
        return res

    def update_penalty(self, psi, z, z0, rho):
        """Update rho for a faster convergence"""
        r = cp.linalg.norm(psi - z)**2
        s = cp.linalg.norm(rho*(z-z0))**2
        if (r > 10*s):
            rho *= 2
        elif (s > 10*r):
            rho *= 0.5
        return rho

    def grad_ptycho(self, data, psi, prb, scan, zlamd, rho, niter):
        """Gradient solver for the ptychography problem |||FQpsi|-sqrt(data)||^2_2 + rho||psi-zlamd||^2_2"""
        # minimization functional
        def minf(fpsi, psi):
            f = cp.linalg.norm(cp.abs(fpsi) - cp.sqrt(data))**2
            if(rho > 0):
                f += rho*cp.linalg.norm(psi-zlamd)**2
            return f

        for i in range(niter):
            # compute the gradient
            fpsi = self.fwd_ptycho(psi, prb, scan)

            gradpsi = self.adj_ptycho(
                fpsi - cp.sqrt(data)*fpsi/(cp.abs(fpsi)+1e-32), prb, scan)

            # normalization coefficient for skipping the line search procedure
            afpsi = self.adj_ptycho(fpsi, prb, scan)
            norm_coeff = cp.real(cp.sum(psi*cp.conj(afpsi)) /
                                 (cp.sum(afpsi*cp.conj(afpsi))+1e-32))

            if(rho > 0):
                gradpsi += rho*(psi-zlamd)
                gradpsi *= min(1/rho, norm_coeff)/2
            else:
                gradpsi *= norm_coeff/2
            # update psi
            psi = psi - 0.5*gradpsi
            # check convergence
            # print(f'{i}) {minf(fpsi, psi).get():.2e} ')

        return psi

    def grad_ptycho_batch(self, data, psi, prb, scan, zlamd, rho, piter):
        """Gradient solver with splitting by nodes"""
        for k in range(self.nnodes):
            ids = cp.arange(k*self.nscan, (k+1)*self.nscan)
            psi[k] = self.grad_ptycho(
                data[ids], psi[k], prb, scan[:, ids], zlamd[k], rho, piter)
        return psi

    def take_lagr(self, data, psi, prb, scan, z, lamd, rho):
        """Compute Lagrangian"""
        lagr = np.zeros(4, dtype='float32')
        for k in range(self.nnodes):
            ids = cp.arange(k*self.nscan, (k+1)*self.nscan)
            lagr[0] += cp.linalg.norm(cp.abs(self.fwd_ptycho(psi[k],
                                      prb, scan[:, ids]))-cp.sqrt(data[ids]))**2
        lagr[1] = 2*cp.sum(cp.real(cp.conj(lamd)*(psi-z)))
        lagr[2] = rho*cp.linalg.norm(psi-z)**2
        lagr[3] = cp.sum(lagr[:3])
        return lagr




    # Regularization todo

    # def fwd_reg(self, psi):
    #     """Forward operator for regularization (J)"""
    #     res = cp.zeros([2, *psi.shape], dtype='complex64')
    #     res[0, :, :-1] = psi[:, 1:]-psi[:, :-1]
    #     res[1, :-1, :] = psi[1:, :]-psi[:-1, :]
    #     res *= 1/np.sqrt(2)  # normalization
    #     return res

    # def adj_reg(self, gr):
    #     """Adjoint operator for regularization (J^*)"""
    #     res = cp.zeros(gr.shape[1:], dtype='complex64')
    #     res[:, 1:] = gr[0, :, 1:]-gr[0, :, :-1]
    #     res[:, 0] = gr[0, :, 0]
    #     res[1:, :] += gr[1, 1:, :]-gr[1, :-1, :]
    #     res[0, :] += gr[1, 0, :]
    #     res *= -1/np.sqrt(2)  # normalization
    #     return res

    # def solve_reg(self, psi, lamd, rho, alpha):
    #     """Solve regularizer problem"""
    #     z = self.fwd_reg(psi) + lamd/rho
    #     # Soft-thresholding
    #     za = cp.sqrt(cp.real(cp.sum(z*cp.conj(z), 0)))
    #     z[:, za <= alpha/rho] = 0
    #     z[:, za > alpha/rho] -= alpha/rho * \
    #         z[:, za > alpha/rho]/(za[za > alpha/rho])
    #     return z
