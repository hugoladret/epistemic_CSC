'''
@hugoladret
> Implements utility tools, notably the local contrast normalise to speed up the pre-processing using Torch rather than Sporco
> Usage : import and call local_contrast_normalise, should be straightforward, no need for extra dimensions
Copied and modified from Wohlberg's SPORCO library
'''

import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # yikes

def local_contrast_normalise(s, n=7, device = None):
    '''
    Local contrast norm of an image, by substraction of the local mean and vision by the local norm.
    Copied from SPORCO (it's better than global normalizations)
    '''
    
    N = 2 * n + 1 # Construct region weighting filter
    g = gaussian_filter2d((N, N), sigma=1.0).to(device)
    sp = symm_pad(s, (n,n,n,n), device = device)

    smn = torch.roll(fftconv(g, sp, device = device), (-n, -n), dims=(0, 1))  # Compute local mean and subtract from image
    s1 = sp - smn
    
    snrm = torch.roll(torch.sqrt(torch.clip(fftconv(g, s1**2, device = device), 0.0, torch.inf)),
                (-n, -n), dims=(0, 1)) # Compute local norm

    snrm = torch.max(torch.mean(snrm, axis=(0, 1), keepdims=True), snrm)
    s2 = s1 / snrm
    
    return s2[n:-n, n:-n]
    #return torch.flip(torch.rot90(s2[n:-n, n:-n],k = 1, dims = [1,0]), dims = [1,]) # this is only a problem with sporco images


def symm_pad(im, padding, device = None):
    '''
    Symmetric padding in PyTorch, from https://discuss.pytorch.org/t/symmetric-padding/19866
    '''
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = torch.arange(-left, w+right)
    y_idx = torch.arange(-top, h+bottom)

    def reflect(x, minx, maxx):
        """ Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length """
        rng = maxx - minx
        double_rng = 2*rng
        mod = torch.fmod(x - minx, double_rng)
        normed_mod = torch.where(mod < 0, mod+double_rng, mod)
        out = torch.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return torch.tensor(out.clone().detach(), dtype = x.dtype)

    x_pad = reflect(x_idx, -0.5, w-0.5)
    y_pad = reflect(y_idx, -0.5, h-0.5)
    xx, yy = torch.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]


def fftconv(a, b, device = None):
    '''
    Torch doesn't support fftconv yet, so this is ALSO copied from SPORCO
    '''

    axes = tuple(range(a.ndim))
    dims = torch.max(torch.tensor([a.shape[i] for i in axes], device = device),
                     torch.tensor([b.shape[i] for i in axes], device = device))
    af = torch.fft.rfftn(a, tuple(dims), axes)
    bf = torch.fft.rfftn(b, tuple(dims), axes)
    ab = torch.fft.irfftn(af * bf, tuple(dims), axes)

    return ab

def gaussian_filter2d(shape, sigma=1.0):
    """ Sample a 2D Gaussian pdf, normalized to have unit sum.
    Parameters
    ----------
    shape : tuple
      Shape of the output tensor.
    sigma : float, optional (default 1.0)
      Standard deviation of the Gaussian pdf.
    Returns
    -------
    gc : torch.Tensor
      Sampled 2D Gaussian pdf.
    """

    if isinstance(shape, int):
        shape = (shape, shape)
    x = torch.linspace(-3.0, 3.0, shape[0])
    y = torch.linspace(-3.0, 3.0, shape[1])
    x, y = torch.meshgrid(x, y)
    gc = torch.exp(-(x**2 + y**2) / (2.0 * sigma**2)) / (2.0 * torch.pi * sigma**2)
    gc /= gc.sum()
    return gc
  
  
# Setting up some functions 
def psnr(vref, vcmp, rng=None):
    dv = (torch.abs(vref.max() - vref.min()))**2
    mse_val = torch.mean((vref - vcmp) ** 2)
    rt = dv / mse_val
    return 10.0 * torch.log10(rt)
