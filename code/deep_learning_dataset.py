import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import time 
import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import torch_cbpdn as cbpdn
import deep_learning_utils as utils

import warnings
warnings.filterwarnings("ignore") # complex32 warning otherwise


# ---------------------------------
# METHODS SC-----------------------
# ---------------------------------
@torch.no_grad()
def run_encoding(D, nscale,cbpdn_params, device,batch_size, workers) :
    # ---------------------------------
    # RUNNING SPARSE CODING -----------
    # ---------------------------------
    if torch.cuda.is_available() :
        device = torch.device('cuda') 
    else :
        device = torch.device('cpu')
        
    # Load dictionary
    D = torch.tensor(D, dtype = torch.float32, device = device)

    # Get CIFAR-10 Datasets
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


    # And run for train
    if not os.path.exists('./data/cifar_sparse/X_train/') :
        os.makedirs('./data/cifar_sparse/X_train/') 
        encode_dataset(D = D, cbpdn_params=cbpdn_params, 
                    device = device, loader = train_loader, nscale = nscale, 
                    desc = 'Training set', savepath = 'train')
        torch.cuda.empty_cache()
    else : 
        print('Folder already exists, skipping encoding of training set')

    # And run for val
    if not os.path.exists('./data/cifar_sparse/X_val/') :
        os.makedirs('./data/cifar_sparse/X_val/') 
        encode_dataset(D = D, cbpdn_params=cbpdn_params, 
                    device = device, loader = val_loader, nscale = nscale, 
                    desc = 'Validation set', savepath = 'val')
        torch.cuda.empty_cache()
    else : 
        print('Folder already exists, skipping encoding of validation set')

    torch.cuda.empty_cache()
    del train_loader, val_loader 


@torch.no_grad()
def encode_dataset(D, cbpdn_params, loader, 
                    nscale, device,
                    desc,
                    savepath) :
    
    modded_data = torch.empty((loader.dataset.data.shape[0], 
                            loader.dataset.data.shape[1]*nscale,
                            loader.dataset.data.shape[2]*nscale),
                            device = device)
    
    psnrs = torch.zeros(modded_data.shape[0], device = device)
    sparsenesses = torch.zeros_like(psnrs)
    
    t_gray = transforms.Grayscale() 
    t_reshape = transforms.Resize((modded_data.shape[1], modded_data.shape[2]))

    begtime = time.time()
    
    for i_img in tqdm(range(modded_data.shape[0]), 'Modifying image %s' % desc) :
        torch.cuda.empty_cache()
        
        D_tensor = torch.tensor(D, dtype = torch.float32, device = device)
        
        img = torch.tensor(loader.dataset.data[i_img,:,:,:], dtype = torch.float32, device = torch.device('cpu'))
        img = torch.swapaxes(img, 0, -1)
        img = t_gray(img)
        img = t_reshape(img)
        img = torch.tensor(img.clone().detach(), dtype = torch.float32, device = device)
        img = img.squeeze(0)
        
        S = utils.local_contrast_normalise(img, device = device)
        
        b = cbpdn.CBPDN(D_tensor, S,**cbpdn_params, device = device)
        X = b.solve()
        reconstructed = b.reconstruct().squeeze()
                
        psnrs[i_img] = utils.psnr(img, reconstructed)
        
        #ranged_reconstructed = (255*(reconstructed - torch.min(reconstructed))/(reconstructed.max() - reconstructed.min()))
        #modded_data[i_img,:,:] = reconstructed

        if torch.all(torch.isnan(reconstructed)) :
            print('NANs in reconstruction whilst reconstructing image %s' % i_img)
            plt.imshow(S.cpu().numpy())
            plt.show()
            raise
        
        # coefficients will be saved one by one, lest we run out of memory
        X = X.squeeze()
        np.save('./data/cifar_sparse/X_%s/%s.npy' % (savepath, i_img),
                X.cpu().numpy())
        
        del D_tensor, img, S, b, reconstructed, X
        
    #np.save('./data/cifar_sparse/%s.npy' % savepath, modded_data.cpu().numpy())
    np.save('./data/cifar_sparse/%s_psnrs.npy' % savepath, psnrs.cpu().numpy())
    np.save('./data/cifar_sparse/%s_sparsenesses.npy' % savepath, sparsenesses.cpu().numpy())
    
    print('Time elapsed : %s' % (time.time() - begtime))
    print('Mean time per image : %s' % ((time.time() - begtime)/modded_data.shape[0]))