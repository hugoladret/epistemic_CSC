import os
import numpy as np
import imageio 
import matplotlib.pyplot as plt 
from met_brewer import met_brew

from tqdm import tqdm 

import torch 
import torchvision.transforms as transforms

#import torch_cbpdn as cbpdn
from sporco.admm import cbpdn

#from sporco.admm import cbpdn
from skimage.metrics import structural_similarity as ssim
from sporco import signal, metric
from skimage import color
from joblib import Parallel, delayed

import sporco.metric as sm
    
    
@torch.no_grad()
def do_cbpdn_dataset(dataset, #patch_sizes,
                    D, cbpdn_params, device, #restart = False,
                    savepath = 'coeffs') :
    # This creates smaller image patches
    sub_datasets = []
    for impath in tqdm(dataset, total = len(dataset), desc = 'Reloading images . . .') :
        img = imageio.imread(impath)
        sub_datasets.append(img)
        
    sub_datasets = np.array(sub_datasets)
    print('Shape of the image patches to sparse code :' )
    print(sub_datasets.shape)
    
    # And this does the sparse coding
    psnrs = np.zeros(len(sub_datasets))
    sparsenesses = np.zeros_like(psnrs)
    t_gray = transforms.Grayscale() 
    
    for i, img in tqdm(enumerate(sub_datasets), desc = 'Sparse Coding images . . .', total = sub_datasets.shape[0]) :
        '''torch.cuda.empty_cache()
        
        D_tensor = torch.tensor(D, dtype = torch.float64, device = device)
        
        img = torch.tensor(img, dtype = torch.float64, device = torch.device('cpu'))
        img = torch.swapaxes(img, 0, -1)
        img = t_gray(img)
        img = torch.tensor(img.clone().detach(), dtype = torch.float64, device = device)
        img = img.squeeze(0)
        
        S = local_contrast_normalise(img, device = device)
        b = cbpdn.CBPDN(D_tensor, S, **cbpdn_params, device = device)
        X = b.solve()
        reconstructed = b.reconstruct().squeeze()
                

        if torch.all(torch.isnan(reconstructed)) :
            print('NANs in reconstruction whilst reconstructing image %s' % i)
            #plt.imshow(S.cpu().numpy(), cmap = 'gray')
            #plt.colorbar()
            #plt.show()
            
        # coefficients will be saved one by one, lest we run out of memory
        X = X.squeeze()
        np.save('./data/%s/%s.npy' % (savepath, i),
                X.cpu().numpy())
        
        psnrs[i] = metric.psnr(S.cpu().numpy(), reconstructed.cpu().numpy())
        sparsenesses[i] = 1 - np.count_nonzero(X.cpu().numpy()) / X.cpu().numpy().size
        
        del D_tensor, img, S, b, reconstructed, X'''
        D = np.float32(D) # CBPDN needs float32
        img = torch.tensor(img, dtype = torch.float64, device = torch.device('cpu'))
        img = torch.swapaxes(img, 0, -1)
        img = t_gray(img)
        img = torch.tensor(img.clone().detach(), dtype = torch.float64, device = device)
        img = img.squeeze(0)
        
        S = local_contrast_normalise(img, device = device)
        white = np.float32(S.cpu().numpy())
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 100,
                                    'RelStopTol': 1e-4, 'AuxVarObj': False})
        b = cbpdn.ConvBPDN(D, white, cbpdn_params['lmbda'], opt, dimK=0)
        X = b.solve()
        reconstructed = b.reconstruct().squeeze()
        X = X.squeeze()
        psnrs[i] = sm.psnr(white, reconstructed)
        sparsenesses[i] = 1 - np.count_nonzero(X) / X.size
        
        os.makedirs(f'./data/{savepath}', exist_ok=True)
        np.save(f'./data/{savepath}/{i}.npy', X)
        
    #np.save('./data/cifar_sparse/%s.npy' % savepath, modded_data.cpu().numpy())
    np.save('./data/%s/psnrs.npy' % savepath, psnrs)
    np.save('./data/%s/sparsenesses.npy' % savepath, sparsenesses)

def make_psnr_sparseness_plots(loadpath = 'coeffs') :
    psnrs = np.load('./data/%s/psnrs.npy' % loadpath)
    psnrs = psnrs[~np.isnan(psnrs)]
    sparsenesses = np.load('./data/%s/sparsenesses.npy' % loadpath)
    sparsenesses = sparsenesses[~np.isnan(sparsenesses)]
    print('PSNRS : median = %s, std = %s, min = %s, max = %s' % (np.median(psnrs), np.std(psnrs), np.min(psnrs), np.max(psnrs)))
    print('Sparsenesses : median = %s, std = %s, min = %s, max = %s' % (np.median(sparsenesses), np.std(sparsenesses), np.min(sparsenesses), np.max(sparsenesses)))
    
    colors = met_brew(name = 'Demuth', n = 2, brew_type='continuous')[::-1]
    for idata, data in enumerate([psnrs, sparsenesses]) :
        fig, ax = plt.subplots(figsize = (5,5))
        
        ax.hist(data, bins = np.linspace(np.percentile(data, 5), np.percentile(data, 95), 25),
                edgecolor = 'w', facecolor = colors[idata])
        ax.axvline(np.median(data), lw = 2, linestyle = '--', c = 'gray')
        
        ax.set_xlim(np.percentile(data, 5), np.percentile(data, 95))
        ax.set_ylim(0, 1.1*np.max(np.histogram(data, bins = np.linspace(np.percentile(data, 5), np.percentile(data, 95), 25))[0]))
        ax.set_xticks(np.linspace(np.percentile(data, 5), np.percentile(data, 95), 5))
        ax.set_yticks(np.linspace(0, 1.1*np.max(np.histogram(data, bins = np.linspace(np.percentile(data, 5), np.percentile(data, 95), 25))[0]), 5, dtype = int))
        
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('PSNR' if idata == 0 else 'Sparseness')
        ax.set_ylabel('# Images')
        
        fig.tight_layout()
        fig.savefig('./figs/psnr_sparseness_%s_%s.pdf' % ('psnr' if idata == 0 else 'sparseness', loadpath), dpi = 300, bbox_inches = 'tight')
    


# ---- Sporco like tools ---- 
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


def psnr(vref, vcmp, rng=None):
    dv = (torch.abs(vref.max() - vref.min()))**2
    mse_val = torch.mean((vref - vcmp) ** 2)
    rt = dv / mse_val
    return 10.0 * torch.log10(rt)

def generate_gaussian_mask(image, sigma=0.5):
    """Generate a Gaussian mask for an image."""
    height, width = image.shape[-2:]

    # Generate a grid of coordinates
    y, x = torch.meshgrid(torch.arange(0., height), torch.arange(0., width))

    # Normalize the coordinates to be in the range [-1, 1]
    y = (y / height) * 2 - 1
    x = (x / width) * 2 - 1

    # Compute the radial distance from the center (Pythagorean theorem)
    dist = torch.sqrt(x ** 2 + y ** 2)

    # Apply the Gaussian function
    mask = torch.exp(- dist ** 2 / (2 * sigma ** 2))

    return mask.to(image.device)