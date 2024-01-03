'''
@hugoladret
> Implements the Convolutional Basis Pursuit DeNoising (CBPDN) algorithm using ADMM methods
> Usage : import CBPDN and call solve() to get sparse coefficients, then reconstruct() to get the denoised image
Copied and modified from Wohlberg's SPORCO library
'''

import torch

@torch.no_grad()
class CBPDN(object) :
    def __init__(self, D, S, device,
                MaxMainIter = 200, lmbda = 0.01, RelStopTol = 0.15, RelaxParam = 0.8,
                L1Weight = 0.5, AutoRho_RsdlRatio = 1.05,
                dimK=None, dimN=2,
                do_ceil = False, ceil = None):
                    
        self.do_ceil = do_ceil 
        self.ceil = ceil  
                    
        # Array shaping
        self.dimN = dimN # number of dimensions for the filters
        self.Nv = S.shape[0:self.dimN]
        self.axisN = tuple(range(0, self.dimN))

        # Instantiating and aliasing
        self.MaxMainIter = MaxMainIter
        self.dtype = S.dtype
        self.device = device
        self.lmbda = torch.as_tensor(lmbda, dtype = self.dtype, device = self.device)
        self.wl1 = torch.as_tensor(L1Weight, dtype = self.dtype, device = self.device)
        self.rho_mu = torch.as_tensor(AutoRho_RsdlRatio, dtype = self.dtype, device = self.device) # autorho mu>xi case, residual ratio between rho and res
        self.rlx = torch.as_tensor(RelaxParam, dtype = self.dtype, device = self.device)
        self.RelStopTol = torch.as_tensor(RelStopTol, dtype = self.dtype, device = self.device)

        # ADMM variables
        self.X = None
        self.Y = torch.empty(self.Nv + (1, 1, D.shape[-1],), dtype=self.dtype, device = self.device)
        self.U = torch.empty_like(self.Y, dtype = self.dtype, device = self.device)

        # Dictionary and Source variables
        self.D = D.view(D.shape[0:self.dimN] + (1, 1, D.shape[-1],))
        self.S = S.view(self.Nv + (1, 1, 1,))

        # Fourier variables
        self.YU = torch.empty(self.Y.shape, dtype=self.dtype, device=self.device)
        self.Df = torch.fft.rfftn(self.D, self.Nv, self.axisN)
        self.DSf = torch.conj(self.Df) * torch.fft.rfftn(self.S, None, self.axisN) 
        ashp = list(self.Y.shape)
        raxis = self.axisN[-1]
        ashp[raxis] = ashp[raxis] // 2 + 1
        self.Xf = torch.empty(ashp, dtype=torch.complex64, device=self.device)

        # Initializing rho and rho_xi
        self.rho = 50.0 * self.lmbda.clone().detach() + 1.0
        self.rho_xi = torch.tensor(float((1.0 + (18.3)**(torch.log10(self.lmbda) + 1.0))),
                                dtype=self.dtype, device=self.device) # Wohlberg 2015 adaptive section 6C

        # Reporting stats
        self.itstat = []

    # inner proudct of two arrays
    def inner(self, x, y, axis=-1):
        xr = x.transpose(axis, 0)
        yr = y.transpose(axis, 0)
        #ip = torch.einsum(xr, [0, ...], yr, [0,...])
        ip = (xr * yr).sum(dim=0).unsqueeze(0)
        return ip.transpose(0, axis)

    def solve(self):

        for self.k in range(self.MaxMainIter):

            # X step
            self.YU[:] = self.Y - self.U
            b = self.DSf + self.rho * torch.fft.rfftn(self.YU, None, self.axisN)

            a = torch.conj(self.Df)
            c = self.Df / (self.inner(self.Df, a, axis=self.dimN + 2) + self.rho) # takes 60% of the time
            self.Xf[:] = (b - (a * self.inner(c, b, axis=self.dimN + 2))) / self.rho # takes 6% of the time
            self.X = torch.fft.irfftn(self.Xf, self.Nv, self.axisN)
            #print(self.dtype)
            #print(self.X.max())
            # Relax AX
            self.AXnr = self.X
            self.AX = self.rlx*self.X + (1 - self.rlx)*self.Y
            
            # Y step
            self.Yprev = self.Y.clone().detach()
            self.Y = torch.sign(self.AX + self.U) * torch.clamp(torch.abs(self.AX + self.U) - ((self.lmbda / self.rho) * self.wl1),
                                                                min = 0)
            #self.Y = torch.clamp(self.Y, min=0)

            # U step
            self.U += (self.AX - self.Y)
            
            # Computing the residuals
            r = torch.norm(self.AXnr - self.Y)
            s = torch.norm(self.rho * (self.Yprev - self.Y))
            epri = torch.max(torch.norm(self.AX), torch.norm(self.Y)) * self.RelStopTol
            edua = self.rho * torch.norm(self.U) * self.RelStopTol

            self.itstat.append([self.k, r, s, epri, edua, self.rho]) # Stats keeping
            if r < epri and s < edua:
                break
            
            # Rho step
            rhomlt = torch.sqrt(r / (s * self.rho_xi) if r > s * self.rho_xi else
                            (s * self.rho_xi) / r)
            rhomlt = torch.clamp(rhomlt, max=100.) # formerly rho_tau / autorhoscaling
            
            rsf = 1.0
            if s > (self.rho_mu / self.rho_xi) * r:
                rsf = 1.0 / rhomlt
            self.rho *= torch.as_tensor(rsf, dtype = self.dtype, device = self.device)
            self.U /= rsf
        
            self.k += 1

        self.Xf = torch.fft.rfftn(self.X, None, self.axisN)
        self.Sf = torch.sum(self.Df * self.Xf, axis=self.dimN + 2)
        return self.X
    
    # Reconstruction from sparse representation (remove if unused ?)
    def reconstruct(self, X=None):
        if X is None:
            X = self.Y
            
        if self.do_ceil : 
            idxs_thresh = torch.where((X < self.ceil) & (X > -self.ceil))
            X[idxs_thresh] = 0

            
        Xf = torch.fft.rfftn(X, None, self.axisN)
        Sf = torch.sum(self.Df * Xf, axis=self.dimN + 2)
        
        #del self.S, self.Y, self.Yprev, self.U 
        #del self.Df, self.AX, self.YU, self.Xf, self.DSf, self.rho, self.rho_xi
        return torch.fft.irfftn(Sf, self.Nv, self.axisN)