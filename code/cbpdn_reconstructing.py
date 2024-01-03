# This is a tad bit simpler, it's just running the CBPDN over the whole dataset
# and getting metrics 

# Luckily we already ported that code over to the img_processing.py file
import torch
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import numpy as np
import imageio 
from tqdm import tqdm 
import os 

import torch 
import torchvision.transforms as transforms
import torch_cbpdn as cbpdn
from skimage.metrics import structural_similarity as ssim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap

from sporco import util
from sporco import signal
import sporco.metric as sm
from sporco.admm import cbpdn as sporco_cbpdn
from SLIP import Image

import img_processing as ip
import scipy.stats as stats

import seaborn as sns
from LogGabor import LogGabor
from lmfit import Model, Parameters

import cbpdn_learning as learning


@torch.no_grad()
def reconstruct_from_img(database_path, D, cbpdn_params, device, savepath) :
    # Creates the savepath folder if it doesnt exist
    if not os.path.exists('./data/%s' % savepath) :
        os.makedirs('./data/%s' % savepath)
        
    dataset = sorted([database_path+'/'+x for x in os.listdir(database_path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
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
        torch.cuda.empty_cache()
        
        from sporco.admm import cbpdn
        D = np.float32(D) # CBPDN needs float32
        img = torch.tensor(img, dtype = torch.float64, device = torch.device('cpu'))
        img = torch.swapaxes(img, 0, -1)
        img = t_gray(img)
        img = torch.tensor(img.clone().detach(), dtype = torch.float64, device = device)
        img = img.squeeze(0)
        
        S = ip.local_contrast_normalise(img, device = device)
        white = np.float32(S.cpu().numpy())
        opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 100,
                                    'RelStopTol': 1e-4, 'AuxVarObj': False})
        b = cbpdn.ConvBPDN(D, white, cbpdn_params['lmbda'], opt, dimK=0)
        X = b.solve()
        reconstructed = b.reconstruct().squeeze()
        X = X.squeeze()
        psnrs[i] = sm.psnr(white, reconstructed)
        sparsenesses[i] = 1 - np.count_nonzero(X) / X.size
        
        np.save('./data/%s/%s.npy' % (savepath, dataset[i].split('/')[-1].split('.')[0]),X)
            
        '''D_tensor = torch.tensor(D, dtype = torch.float64, device = device)
        
        img = torch.tensor(img, dtype = torch.float64, device = torch.device('cpu'))
        img = torch.swapaxes(img, 0, -1)
        img = t_gray(img)
        img = torch.tensor(img.clone().detach(), dtype = torch.float64, device = device)
        img = img.squeeze(0)
        
        S = ip.local_contrast_normalise(img, device = device)
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
        np.save('./data/%s/%s.npy' % (savepath, dataset[i].split('/')[-1].split('.')[0]),
                X.cpu().numpy())
        
        #psnrs[i] = ssim(S.cpu().numpy(), reconstructed.cpu().numpy(), data_range = reconstructed.cpu().numpy().max() - reconstructed.cpu().numpy().min())
        psnrs[i] = sm.psnr(S.cpu().numpy(), reconstructed.cpu().numpy())
        #psnrs[i] = sm.psnr(S.cpu().numpy(), reconstructed.cpu().numpy())
        sparsenesses[i] = 1 - np.count_nonzero(X.cpu().numpy()) / X.cpu().numpy().size
        
        del D_tensor, img, S, b, reconstructed, X'''
        
    #np.save('./data/cifar_sparse/%s.npy' % savepath, modded_data.cpu().numpy())
    np.save('./data/%s/psnrs.npy' % savepath, psnrs)
    np.save('./data/%s/sparsenesses.npy' % savepath, sparsenesses)
    
    
def psnr_sparseness_plot() :
    archs = ['full', 'thin', 'learned', 'learned_thin', '12x12x108']
    dict_colors = ['#F8766D', '#00BFC4', '#00A087', '#3C5488', 'gray'] 
    
    PSNR_min = 22
    PSNR_max = 58
    sparseness_min = 0.985
    sparseness_max = 0.999
    n_bins = 100
    
    # PSNR (or SSIM)
    fig, ax = plt.subplots(1, 1, figsize = (8,5))
    all_psnrs = []
    for i, arch in enumerate(archs) :
        fname = './data/%s/psnrs.npy' % arch
        psnr = np.load(fname)
        
        ax.hist(psnr, bins=np.linspace(PSNR_min, PSNR_max, n_bins), alpha=.5, label=arch, facecolor=dict_colors[i], edgecolor = 'k',
                zorder = 10 if arch == 'online' else i)
        #bins,_ = np.histogram(psnr, bins = np.linspace(0.5, 1., 64))
        all_psnrs.append(psnr)
        ax.vlines(np.median(psnr), 0, 150, ls='--', color=dict_colors[i])
        
    ax.set_xticks([PSNR_min, (PSNR_max+PSNR_min)/2, PSNR_max])
    ax.set_yticks([0, 50, 100, 150])
    ax.set_xlim(PSNR_min, PSNR_max)
    ax.set_ylim(0, 150)
    ax.tick_params(axis='both', which='major', labelsize = 14)

    ax.set_xlabel('PSNR', fontsize = 18)
    ax.set_ylabel('# images', fontsize = 18)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('./figs/fig_2_PSNR.pdf', bbox_inches = 'tight', transparent = True, dpi = 200)
    
    i = 0 # some stats
    print('Comparing stats on PSNR')
    print(archs[i] + ' vs ' + archs[i+1])
    print(stats.mannwhitneyu(all_psnrs[i], all_psnrs[i+1], alternative = 'less'))
    i = 1
    print(archs[i] + ' vs ' + archs[i+1])
    print(stats.mannwhitneyu(all_psnrs[i], all_psnrs[i+1], alternative = 'less'))
    i = 2
    print(archs[i] + ' vs ' + archs[i+1])
    print(stats.mannwhitneyu(all_psnrs[i], all_psnrs[i+1], alternative = 'less'))
    i = 0
    print(archs[i] + ' vs ' + archs[i+4])
    print(stats.mannwhitneyu(all_psnrs[i], all_psnrs[i+4], alternative = 'less'))
    i = 3
    print(archs[i] + ' vs ' + archs[i+1])
    print(stats.mannwhitneyu(all_psnrs[i], all_psnrs[i+1], alternative = 'less'))
    
    
    # Sparsenesses
    fig, ax = plt.subplots(1, 1, figsize = (8,5))
    all_sparsenesses = []
    for i, arch in enumerate(archs) :
        fname = './data/%s/sparsenesses.npy' % arch
        nz = np.load(fname)
        print(nz.min(), nz.max())
        ax.hist(nz, bins=np.linspace(sparseness_min, sparseness_max, n_bins),
                alpha=.5, label=arch, facecolor=dict_colors[i], edgecolor = 'k')
        #bins,_ = np.histogram(nz, bins = np.linspace(0.992, 1.0, 128))
        all_sparsenesses.append(nz)
        ax.vlines(np.median(nz), 0, 100, ls='--', color=dict_colors[i])

    ax.set_xticks([sparseness_min, (sparseness_max+sparseness_min)/2, sparseness_max])
    ax.set_yticks([0, 33, 66, 100])
    ax.set_xlim(sparseness_min, sparseness_max)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='both', which='major', labelsize = 14)

    ax.set_xlabel('Sparseness', fontsize = 18)
    ax.set_ylabel('# images', fontsize = 18)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('./figs/fig_2_sparseness.pdf', bbox_inches = 'tight', transparent = True, dpi = 200)

    # Some more stats 
    print('Comparing stats on sparseness')
    i = 0
    print(archs[i] + ' vs ' + archs[i+1])
    print(stats.mannwhitneyu(all_sparsenesses[i], all_sparsenesses[i+1], alternative = 'less'))
    i = 1
    print(archs[i] + ' vs ' + archs[i+1])
    print(stats.mannwhitneyu(all_sparsenesses[i], all_sparsenesses[i+1], alternative = 'greater'))
    i = 3
    print(archs[i] + ' vs ' + archs[i+1])
    print(stats.mannwhitneyu(all_sparsenesses[i], all_sparsenesses[i+1], alternative = 'greater'))
    i = 0
    print(archs[i] + ' vs ' + archs[i+4])
    print(stats.mannwhitneyu(all_sparsenesses[i], all_sparsenesses[i+4], alternative = 'less'))
    i = 3
    print(archs[i] + ' vs ' + archs[i+1])
    print(stats.mannwhitneyu(all_sparsenesses[i], all_sparsenesses[i+1], alternative = 'greater'))

    # And the 2D plot
    fig, ax = plt.subplots(1, 1, figsize = (8,8))
    for i, arch in enumerate(archs):
        psnr = np.load('./data/%s/psnrs.npy' % arch)
        nz = np.load('./data/%s/sparsenesses.npy' % arch)
        
        ax.scatter(nz, psnr, alpha=.3, label=arch, color = dict_colors[i], s=8,
                zorder = 10 if arch == 'online' else i)

    ax.set_xlim(sparseness_min, sparseness_max)
    ax.set_ylim(PSNR_min, PSNR_max)
    ax.set_xticks([sparseness_min, (sparseness_max+sparseness_min)/2, sparseness_max])
    ax.set_yticks([PSNR_min, (PSNR_max-PSNR_min)/2, PSNR_max])
    ax.tick_params(axis='both', which='major', labelsize = 14)
    ax.set_xlabel('Sparseness', fontsize = 18)
    ax.set_ylabel('PSNR', fontsize = 18)
    ax.legend(loc='best')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.savefig('./figs/fig3_psnr_sparseness.pdf', bbox_inches = 'tight', transparent = True, dpi = 200)
    fig.show()
    
    
def coeff_kde(init_dict, learned_dict, N_theta, N_phase, N_Btheta,
            figname = 'online') :
    
    thetas = np.linspace(0, np.pi, N_theta, endpoint = False)
    phases = np.linspace(0, np.pi, N_phase, endpoint=False)
    B_thetas = np.linspace(0, np.pi/6, N_Btheta+1, endpoint = True)[1:]
    '''if figname == 'learned_thin' :
        B_thetas = np.linspace(0, np.pi/8, N_Btheta+1, endpoint = True)[1:]'''
    
    parameterfile = 'https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py'
    lg = LogGabor(parameterfile)
    filter_size = init_dict.shape[0]
    lg.set_size((filter_size, filter_size))
    
    # Initial parameters ---
    # Re-get the parameters
    LG_params = []
    for i_theta in range(N_theta):
        for i_Btheta in range(N_Btheta):
            for i_phase in range(N_phase):
                params= {'sf_0':.4, 'B_sf': lg.pe.B_sf, 'theta':thetas[i_theta], 'B_theta': B_thetas[i_Btheta],
                        'phase' : phases[i_phase]}
                LG_params.append(params)

    # Refits ---
    errs_theta, errs_B_theta = np.zeros(init_dict.shape[-1]), np.zeros(init_dict.shape[-1])
    new_theta, new_btheta = np.zeros(init_dict.shape[-1]), np.zeros(init_dict.shape[-1])
    new_sf, new_bsf = np.zeros(init_dict.shape[-1]), np.zeros(init_dict.shape[-1])
    r2s = np.zeros(init_dict.shape[-1])
    for i in tqdm(range(learned_dict.shape[-1]), 'Refitting') :
        theta_init = LG_params[i]['theta']
        B_theta_init = LG_params[i]['B_theta']
        filt = norm_data(learned_dict[:,:,i])
        idxs_removes = np.where((filt < 0.15) & (filt > -0.15))
        filt[idxs_removes] = 0
        
        try :
            best_vals, r2 = fit_lg(filt, theta_init = theta_init, B_theta_init = B_theta_init,
                                    B_thetas = B_thetas, thetas = thetas,
                                    phase_init = LG_params[i]['phase'], vary_theta = False if figname == 'learned_thin' else True,
                                    lg=lg)
            
            new_theta[i] = best_vals['theta']
            new_btheta[i] = best_vals['B_theta']
            new_sf[i] = best_vals['sf_0']
            new_bsf[i] = best_vals['B_sf']
            errs_theta[i] = best_vals['theta'] - theta_init
            errs_B_theta[i] = best_vals['B_theta'] - B_theta_init
            r2s[i] = r2
        except ValueError :
            pass
        
    
    '''fig, ax = plt.subplots(figsize = (7,6), subplot_kw = dict(projection = 'polar'))
    kde = sns.kdeplot(x=new_theta, y=new_btheta, levels = 8, color = 'k')
    kde = sns.kdeplot(x=new_theta, y=new_btheta, fill = True, cmap = "magma", cbar = True, levels = 8, color = 'k')
    theta_bins = np.linspace(thetas.min(), np.pi, 5)
    btheta_bins = np.linspace(0.08726646, 0.52359878, 5)

    ax.set_xticks(theta_bins)
    ax.set_xticklabels(np.round(theta_bins*180/np.pi, 1))
    ax.set_yticks(btheta_bins)
    ax.set_yticklabels(np.round(btheta_bins*180/np.pi, 1))

    ax.tick_params(axis='both', which='major', labelsize = 14)

    ax.set_xlim(thetas.min(), np.pi)
    ax.set_ylim(0.08726646, 0.52359878)

    ax.set_xlabel(r'$\theta$', fontsize = 18)
    ax.set_ylabel(r'$B_\theta$', fontsize = 18)'''
    
    # Recreate the figure with the specified adjustments for KDE plot
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.5)

    # KDE plot instead of scatter plot
    main_ax = fig.add_subplot(grid[:-1, 1:])
    sns.kdeplot(x=new_theta, y=new_btheta, ax=main_ax, cmap="Greys")
    #main_ax.set_xlabel('Theta')
    #main_ax.set_ylabel('B_theta')
    main_ax.set_xlim(new_theta.min(), new_theta.max())
    main_ax.set_ylim(new_btheta.min(), new_btheta.max())
    main_ax.set_xticks(np.linspace(new_theta.min(), new_theta.max(), 5))
    main_ax.set_yticks(np.linspace(new_btheta.min(), new_btheta.max(), 5))
    main_ax.set_xticklabels([])
    main_ax.set_yticklabels([])

    # Histogram for new_theta
    hist_x_ax = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    counts, bins = np.histogram(new_btheta, bins=np.linspace(new_btheta.min(), new_btheta.max(), 15))
    for i in range(len(bins) - 1):
        hist_x_ax.barh(bins[i], counts[i], height=bins[i+1]-bins[i], color=plt.cm.magma(i / len(bins)))
    hist_x_ax.invert_xaxis()
    

    # Histogram for new_btheta
    hist_y_ax = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
    counts, bins = np.histogram(new_theta, bins=np.linspace(new_theta.min(), new_theta.max(), 15))
    for i in range(len(bins) - 1):
        hist_y_ax.bar(bins[i], counts[i], width=bins[i+1]-bins[i], color=plt.cm.viridis(i / len(bins)))
    hist_y_ax.invert_yaxis()

    #plt.show()

    fig.tight_layout()
    fig.savefig('./figs/fig3_kde_%s_polar.pdf' % figname, bbox_inches = 'tight', transparent = True, dpi = 200)

    
def reconstruction_plot(D, database_path, coeff_path, images_paths,
                        filename,
                        N_X, N_Y, N_theta, N_Btheta, N_phase,
                        cbpdn_params,
                        cutoffs = None, init_recompute = False) :
    print('Doing reconstruction plot for %s' % filename)
    
    t_gray = transforms.Grayscale() 
    
    # This is where we plot the reconstructions
    # First we need to reverse lookup the coefficients idxs from the image paths
    #dataset = sorted([database_path+'/'+x for x in os.listdir(database_path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # find the indices of images_paths 
    coeffs = []
    for impath in images_paths :
        if init_recompute : 
            img = imageio.imread(database_path + '/' +impath)
            img = img[::8, ::8, :]
            
            N_X, N_Y = img.shape[0], img.shape[1]
                
                
            from sporco.admm import cbpdn
            D = np.float32(D) # CBPDN needs float32
            img = torch.tensor(img, dtype = torch.float64, device = torch.device('cpu'))
            img = torch.swapaxes(img, 0, -1)
            img = t_gray(img)
            img = torch.tensor(img.clone().detach(), dtype = torch.float64, device = 'cuda')
            img = img.squeeze(0)
            
            S = ip.local_contrast_normalise(img, device = 'cuda')
            white = np.float32(S.cpu().numpy())
            opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 100,
                                        'RelStopTol': 1e-4, 'AuxVarObj': False})
            b = cbpdn.ConvBPDN(D, white, cbpdn_params['lmbda'], opt, dimK=0)
            X = b.solve()
            reconstructed = b.reconstruct().squeeze()
            print('Reconstructed the big image.')
            
            '''D_tensor = torch.tensor(D, dtype = torch.float64, device = 'cuda')
    
            img = torch.tensor(img, dtype = torch.float64, device = torch.device('cpu'))
            img = torch.swapaxes(img, 0, -1)
            img = t_gray(img)
            img = torch.tensor(img.clone().detach(), dtype = torch.float64, device = 'cuda')
            img = img.squeeze(0)
            
            S = ip.local_contrast_normalise(img, device = 'cuda')
            b = cbpdn.CBPDN(D_tensor, S, **cbpdn_params, device = 'cuda')
        
            X = b.solve()'''
            this_coeff = X.reshape((N_X, N_Y, N_theta, N_Btheta, N_phase))
            this_coeff = this_coeff.sum(axis = (4))
            coeffs.append(this_coeff)
        else :
            thispath = impath.split('.')[0]
            this_coeff = np.load('./data/%s/%s.npy' % (coeff_path, thispath))
            #this_coeff = this_coeff[np.abs(this_coeff) > 0]
            this_coeff = this_coeff.reshape((N_X, N_Y, N_theta, N_Btheta, N_phase))

            this_coeff = this_coeff.sum(axis=(4))
            coeffs.append(this_coeff)
    
    
    # check if cutoffs are not null 
    if cutoffs is not None :
        cmap = plt.get_cmap('twilight')
        for icutoff in range(len(cutoffs)):
            for i_bt in range(N_Btheta):
                for i in range(len(images_paths)):
                    fig, ax = plt.subplots(figsize=(5, 5), ncols=1)
                    im_RGB = np.zeros((N_X, N_Y, 3))
                    for i_theta, theta_ in enumerate(np.linspace(0, 180, N_theta, endpoint=False)):
                        one_coeff = np.absolute(coeffs[i][:, :, i_theta, i_bt])
                        # this is not necessary because we are shifting the coefficients to their absolute value
                        idxs_thresh = np.where((one_coeff < cutoffs[icutoff]) & (one_coeff > -cutoffs[icutoff]))
                        one_coeff[idxs_thresh] = 0
                        
                        im_abs = 1. * np.flipud(np.fliplr(np.abs(one_coeff)))
                        
                        # Convert orientation into grayscale
                        grayscale_intensity = .5 * np.sin(2 * theta_) + .5
                        RGB = np.asarray(cmap(grayscale_intensity)[:3])
                        
                        im_RGB += im_abs[:, :, np.newaxis] * RGB[np.newaxis, np.newaxis, :]

                    im_RGB /= im_RGB.max()
                    im_RGB = np.flip(im_RGB, axis=(0, 1))
                    ax.imshow(im_RGB, interpolation='none')
                    ax.axis('off')

                fig.tight_layout()
                fig.savefig('./figs/fig4_reconstructions_coeffs_%s_bt%s_cutoff%s.pdf' % (filename, i_bt, icutoff),
                            bbox_inches='tight', transparent=True, dpi=200)
                plt.close(fig) 

                
    else :
        cmap = plt.get_cmap('twilight')
        for i_bt in range(N_Btheta):
            for i in range(len(images_paths)):
                fig, ax = plt.subplots(figsize=(5, 5), ncols=1)
                im_RGB = np.zeros((N_X, N_Y, 3))
                for i_theta, theta_ in enumerate(np.linspace(0, 180, N_theta, endpoint=False)):
                    one_coeff = np.absolute(coeffs[i][:, :, i_theta, i_bt])
                    
                    im_abs = 1. * np.flipud(np.fliplr(np.abs(one_coeff)))
                    
                    # Convert orientation into grayscale
                    grayscale_intensity = .5 * np.sin(2 * theta_) + .5
                    RGB = np.asarray(cmap(grayscale_intensity)[:3])
                    
                    im_RGB += im_abs[:, :, np.newaxis] * RGB[np.newaxis, np.newaxis, :]

                im_RGB /= im_RGB.max()
                im_RGB = np.flip(im_RGB, axis=(0, 1))
                ax.imshow(im_RGB, interpolation='none')
                ax.axis('off')

                fig.tight_layout()
                fig.savefig('./figs/fig4_reconstructions_coeffs_%s_%s_bt%s.pdf' % (filename, i, i_bt),
                            bbox_inches='tight', transparent=True, dpi=200)
                plt.close(fig)  
                
            del im_RGB, im_abs, RGB, grayscale_intensity

    coeffs = [] # UNUSED AFTERWARDS EXCEPT FOR ENUMERATION PURPOSES
    for impath in images_paths :
        thispath = impath.split('.')[0]
        this_coeff = np.load('./data/%s/%s.npy' % (coeff_path, thispath))
        #this_coeff = this_coeff.reshape((N_X, N_Y, N_theta, N_Btheta, N_phase))
        # Zero out values that don't meet the condition
        #this_coeff = np.where(np.abs(this_coeff) > 0, this_coeff, 0)

        #this_coeff = this_coeff.sum(axis=(4))

        coeffs.append(this_coeff)
        
    
    # Now we do some reconstruction 
    if not init_recompute : 
        if cutoffs is not None :
            for icutoff in range(len(cutoffs)) :
                for icoeff, coeff in enumerate(coeffs) :
                    torch.cuda.empty_cache()
                    #idx_img = idxs[icoeff]
                    
                    #img = util.ExampleImages(pth = './').image(fname = dataset[idx_img], scaled=True, gray=True)
                    img = imageio.imread(database_path + '/' +images_paths[icoeff].split('.')[0]+'.jpg')
                    
                    from sporco.admm import cbpdn
                    D = np.float32(D) # CBPDN needs float32
                    img = torch.tensor(img, dtype = torch.float64, device = torch.device('cpu'))
                    img = torch.swapaxes(img, 0, -1)
                    img = t_gray(img)
                    img = torch.tensor(img.clone().detach(), dtype = torch.float64, device = 'cuda')
                    img = img.squeeze(0)
                    
                    S = ip.local_contrast_normalise(img, device = 'cuda')
                    white = np.float32(S.cpu().numpy())
                    opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 100,
                                                'RelStopTol': 1e-4, 'AuxVarObj': False})
                    b = cbpdn.ConvBPDN(D, white, cbpdn_params['lmbda'], opt, dimK=0)
                    X = b.solve()
                    idxs_thresh = np.where((X < cutoffs[icutoff]) & (X > -cutoffs[icutoff]))
                    X[idxs_thresh] = 0
                    reconstructed = b.reconstruct(X = X).squeeze()
                    
                    #psnrs[i] = sm.psnr(white, reconstructed)
                    #nzs[i] = 1 - np.count_nonzero(X) / X.size
                
                    '''D_tensor = torch.tensor(D, dtype = torch.float64, device = 'cuda')
            
                    img = torch.tensor(img, dtype = torch.float64, device = torch.device('cpu'))
                    img = torch.swapaxes(img, 0, -1)
                    img = t_gray(img)
                    img = torch.tensor(img.clone().detach(), dtype = torch.float64, device = 'cuda')
                    img = img.squeeze(0)
                    
                    S = ip.local_contrast_normalise(img, device = 'cuda')
                    b = cbpdn.CBPDN(D_tensor, S, **cbpdn_params, device = 'cuda')
                
                    X = b.solve()
                    idxs_thresh = torch.where((X < cutoffs[icutoff]) & (X > -cutoffs[icutoff]))
                    X[idxs_thresh] = 0'''
                    
                    '''ceil = torch.max(X)/(1/cutoffs[icutoff])
                    idxs_thresh = torch.where((X < ceil) & (X > -ceil))
                    X[idxs_thresh] = 0'''
                    '''reconstructed = b.reconstruct(X = torch.tensor(X, device = 'cuda')).squeeze()'''
                    
                    if icutoff == 0 or icutoff == 3 or icutoff == 6 :
                        fig, ax = plt.subplots(figsize = (5,5))
                        ax.imshow(reconstructed, cmap = 'gray')
                        ax.axis('off')
                        fig.tight_layout()
                        fig.savefig('./figs/fig4_reconstructions_%s_%s_cutoff%s.pdf'%(filename,icoeff,icutoff), bbox_inches = 'tight', transparent = True, dpi = 200)
                        plt.close(fig)
                    #del img, D_tensor, S, b, X, reconstructed
        else :
            for icoeff, coeff in enumerate(coeffs) :
                torch.cuda.empty_cache()
                img = imageio.imread(database_path + '/' +images_paths[icoeff].split('.')[0]+'.jpg')
                
                '''D_tensor = torch.tensor(D, dtype = torch.float64, device = 'cuda')
            
                img = torch.tensor(img, dtype = torch.float64, device = torch.device('cpu'))
                img = torch.swapaxes(img, 0, -1)
                img = t_gray(img)
                img = torch.tensor(img.clone().detach(), dtype = torch.float64, device = 'cuda')
                img = img.squeeze(0)
                
                S = ip.local_contrast_normalise(img, device = 'cuda')
                b = cbpdn.CBPDN(D_tensor, S, **cbpdn_params, device = 'cuda')
                X = b.solve()
                reconstructed = b.reconstruct().squeeze()'''
                
                from sporco.admm import cbpdn
                D = np.float32(D) # CBPDN needs float32
                img = torch.tensor(img, dtype = torch.float64, device = torch.device('cpu'))
                img = torch.swapaxes(img, 0, -1)
                img = t_gray(img)
                img = torch.tensor(img.clone().detach(), dtype = torch.float64, device = 'cuda')
                img = img.squeeze(0)
                
                S = ip.local_contrast_normalise(img, device = 'cuda')
                white = np.float32(S.cpu().numpy())
                opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 100,
                                            'RelStopTol': 1e-4, 'AuxVarObj': False})
                b = cbpdn.ConvBPDN(D, white, cbpdn_params['lmbda'], opt, dimK=0)
                X = b.solve()
                reconstructed = b.reconstruct(X = X).squeeze()
                    
                
                fig, ax = plt.subplots(figsize = (5,5))
                ax.imshow(reconstructed, cmap = 'gray')
                ax.axis('off')
                fig.tight_layout()
                fig.savefig('./figs/fig4_reconstructions_%s_%s.pdf'%(filename, icoeff), bbox_inches = 'tight', transparent = True, dpi = 200)
                plt.close(fig)
                np.save('./data/%s/reconstructed_%s.npy' % (coeff_path, icoeff), reconstructed)
                #del img, D_tensor, S, b, X, reconstructed
    
    
def compute_img_differences(coeff_path_1, coeff_path_2, coeffs) :
    from matplotlib import colors
    divnorm=colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1)

    for icoeff, _ in enumerate(coeffs) :
        coeff_1 = np.load('./data/%s/reconstructed_%s.npy' % (coeff_path_1, icoeff))
        coeff_2 = np.load('./data/%s/reconstructed_%s.npy' % (coeff_path_2, icoeff))
        
        diff = coeff_1 - coeff_2
        fig, ax = plt.subplots(figsize = (5,5))
        ax.imshow(diff, cmap = 'coolwarm', norm = divnorm)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig('./figs/fig4_reconstructions_diff_%s.pdf' % icoeff, bbox_inches = 'tight', transparent = True, dpi = 200)
        fig.show()


def resilience_plot(dico, database_path, coeff_path, filename,
                    N_X, N_Y, N_theta, N_Btheta, N_phase,
                    cutoffs, colormap,
                    lmbda, cbpdn_params, 
                    device, 
                    tot_images = 100) :
    
    print('Doing resilience plot for %s'%filename)
    
    
    t_gray = transforms.Grayscale() 
    
    cbpdn_params['RelStopTol'] = 1e-2 #speeds up 
    
    dataset = sorted([database_path+'/'+x for x in os.listdir(database_path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])[:tot_images]
    sub_datasets = []
    for impath in tqdm(dataset, total = len(dataset), desc = 'Reloading images . . .') :
        img = imageio.imread(impath)
        sub_datasets.append(img)
    sub_datasets = np.array(sub_datasets)

    fig, ax = plt.subplots(1, 1, figsize = (8,8))
    local_psnrs, local_sparsenesses = [], []
    for iceil, ceil in enumerate(cutoffs) :
        
        # And this does the sparse coding
        psnrs = np.zeros(tot_images)
        nzs = np.zeros(tot_images)
        t_gray = transforms.Grayscale() 
        
        for i, img in tqdm(enumerate(sub_datasets), desc = 'Reconstructing for cutoff %.3f'% ceil,  total = sub_datasets.shape[0]) :
            torch.cuda.empty_cache()
            
            from sporco.admm import cbpdn
            D = np.float32(dico) # CBPDN needs float32
            img = torch.tensor(img, dtype = torch.float64, device = torch.device('cpu'))
            img = torch.swapaxes(img, 0, -1)
            img = t_gray(img)
            img = torch.tensor(img.clone().detach(), dtype = torch.float64, device = device)
            img = img.squeeze(0)
            
            S = ip.local_contrast_normalise(img, device = device)
            white = np.float32(S.cpu().numpy())
            opt = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': 100,
                                        'RelStopTol': 1e-4, 'AuxVarObj': False})
            b = cbpdn.ConvBPDN(D, white, lmbda, opt, dimK=0)
            X = np.load('./data/%s/%s.npy' % (coeff_path, dataset[i].split('/')[-1].split('.')[0]))
            X = np.float32(np.expand_dims(X, (2,3)))
            #X = b.solve()
            idxs_thresh = np.where((X < ceil) & (X > -ceil))
            X[idxs_thresh] = 0 
            
            reconstructed = b.reconstruct(X=X).squeeze()
            
            psnrs[i] = sm.psnr(white, reconstructed)
            nzs[i] = 1 - np.count_nonzero(X) / X.size
            
            ''' D_tensor = torch.tensor(dico, dtype = torch.float64, device = device)
            
            img = torch.tensor(img, dtype = torch.float64, device = torch.device('cpu'))
            img = torch.swapaxes(img, 0, -1)
            img = t_gray(img)
            img = torch.tensor(img.clone().detach(), dtype = torch.float64, device = device)
            img = img.squeeze(0)
            
            S = ip.local_contrast_normalise(img, device = device)
            b = cbpdn.CBPDN(D_tensor, S, **cbpdn_params, device = device, do_ceil = True, ceil = ceil)
            X = b.solve()
            reconstructed = b.reconstruct().squeeze()
            
                    
            if torch.all(torch.isnan(reconstructed)) :
                print('NANs in reconstruction whilst reconstructing image %s' % i)
                #plt.imshow(S.cpu().numpy(), cmap = 'gray')
                #plt.colorbar()
                #plt.show()

            # This is also done in the CBPDN class, but we need to do it here to get the sparsity
            X = X.squeeze()
            idxs_thresh = torch.where((X < ceil) & (X > -ceil))
            X[idxs_thresh] = 0 
            psnrs[i] = sm.psnr(S.cpu().numpy(), reconstructed.cpu().numpy())
            nzs[i] = 1 - np.count_nonzero(X.cpu().numpy()) / X.cpu().numpy().size
            
            del D_tensor, img, S, b, reconstructed, X'''
            
        ax.scatter(nzs, psnrs, alpha=.3, color = colormap[iceil])
        local_psnrs.append(psnrs)
        local_sparsenesses.append(nzs)
            
    #ax.set_xlim(2e-6, 3e-5)
    #ax.set_ylim(.55, 1.0)
    #ax.set_xticks([2e-6, (2e-6+3e-5)/2, 3e-5])
    #ax.set_yticks([0.55, 0.775, 1.])
    
    ax.set_xlim(0.985, 1.)
    ax.set_ylim(0, 60)
    ax.set_xticks([0.985, 0.990, 0.995, 1.000])
    ax.set_yticks([0, 20, 40, 60])
    
    ax.tick_params(axis='both', which='major', labelsize = 14)

    ax.set_xlabel('Sparseness', fontsize = 18)
    ax.set_ylabel('PSNR', fontsize = 18)
    
    # Create an axis for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cmap = ListedColormap(colormap)
    bounds = np.linspace(0, .5, len(colormap)+1)
    norm = plt.Normalize(bounds.min(), bounds.max())
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, ticks=bounds, boundaries=bounds, format='%.2f')
    cb.set_label('Threshold Value', fontsize=14)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.savefig('./figs/fig3_psnr_sparseness_decay_%s.pdf' % filename,
                bbox_inches = 'tight', transparent = True, dpi = 200)
    plt.show(block = False)
    
    return local_psnrs, local_sparsenesses 
    
    
def lg_model(x, theta, B_theta, phase, sf_0, B_sf, normer, filter_size, lg) :
    env = lg.loggabor(filter_size//2, filter_size//2,
                    theta = theta, B_theta = B_theta,
                    sf_0 = sf_0, B_sf = B_sf)
    env *= np.exp(-1j * phase)
    normd = lg.normalize(lg.invert(env)*lg.mask)
    return normd.flatten()/normer

def fit_lg(filt, theta_init, B_theta_init, phase_init, B_thetas, thetas, lg,
        vary_theta = True):
    
    y = filt.flatten()
    x = np.linspace(0, y.shape[0], y.shape[0])
    
    mod = Model(lg_model, independent_vars=['x', 'lg'])

    pars = Parameters()
    pars.add_many(('theta', theta_init, vary_theta, thetas.min(),  thetas.max()),
                ('B_theta', B_theta_init, True, 0.08726646, 0.52359878), # hard coded here to simplify refit of thins
                ('phase', phase_init, False, 0, 1),
                ('sf_0', 0.4, False, 0.1, 1.0), 
                ('B_sf', 0.4, False, 0.1 , 1.0),
                ('normer', 1., False, 4., 0.5),
                ('filter_size', filt.shape[0], False, None, None))

    out = mod.fit(y, x = x, lg = lg,
                params = pars, nan_policy = 'raise', max_nfev = 8000)
    return out.best_values, np.abs(1-out.residual.var() / np.var(y))

# function that normalizes data between -1 and 1
def norm_data(data) :
    return (data - data.min())/(data.max() - data.min())*2 - 1

