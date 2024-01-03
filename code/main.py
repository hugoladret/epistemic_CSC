import numpy as np
import matplotlib.pyplot as plt
import met_brewer
import os 
import torch
from scipy import stats

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import preprocessing as pre
import img_processing as imgp
import coeff_analysis as ca
import cbpdn_paramscan as scan
import cbpdn_learning as learning
import cbpdn_reconstructing as reconstructing

if torch.backends.mps.is_available():
    device = torch.device('mps')
    # HACK: TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
    device = torch.device('cpu')

elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # HACK: Intel MKL Issue on some Windows machines

# --- Parameters --- #
# Data params
data_all = './imgs/all'
data_resized = './imgs/resized'
patch_sizes = (256, 256) # size of the patches on which to do csc

# Dictionary generation 
filter_size = 12
N_theta_thin, N_Btheta_thin, N_phase_thin = 72, 1, 2
N_theta, N_Btheta, N_phase = 12, 6, 2
thin_path = './data/dictionary_thin.npz'
full_path = './data/dictionary.npz'

# CBPDN params
cbpdn_params = {'MaxMainIter' : 100, 'lmbda' : 1e-3, 'RelStopTol' : 1e-3, 'RelaxParam' : 1.8,
                'L1Weight' : 1.0, 'AutoRho_RsdlRatio' : 1.05}

# The ol' switchflip board, use booleans instead of not os statements to switch on/off parts of the code

do_pre = not(os.path.exists(data_resized)) # automagic - remove folder to recompute
do_label = not(os.path.exists('figs/image_labels.pdf')) # automagic - remove folder to recompute
do_CSC = not(os.path.exists('./data/coeffs')) # automagic - remove folder to recompute
do_analysis = True
do_CSC_btheta = not(os.path.exists('./data/btheta')) # automagic - remove folder to recompute

do_analysis_btheta = True 

do_sporco_learning = True
learning_thin = True
learning_bt = True
learning_scratch = True

do_sporco_reconstructing = True
do_sporco_resilience = True
do_sporco_param_scan = True

######################
# ---  Part one  --- #
######################
# --- Preprocessing --- # 
if do_pre :
    # Get all images from the folders 
    dataset = [data_all+'/'+x for x in os.listdir(data_all) if x.lower().endswith(('.png', '.jpg', '.jpeg')) and not 'resized' in x]

    # Run the preprocessing on the data, rescaling and greyscaling
    pre.resize_images(dataset, target_size = patch_sizes)
    
    pre.generate_dicos(filter_size,
                N_theta_thin, N_Btheta_thin, N_phase_thin, thin_path,
                N_theta, N_Btheta, N_phase, full_path)
    
# --- Labelling --- #
if do_label :
    # Describe the dataset using ResNet tags
    pre.label_images(paths = [data_resized+'/'+x for x in os.listdir(data_resized) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])

# --- CBPDN --- #
# Theta only    #
# ------------- #
if do_CSC :
    # Running the CSC on images patches 
    dico = np.load(thin_path)['D']
    imgp.do_cbpdn_dataset(dataset = [data_resized+'/'+x for x in os.listdir(data_resized) if x.lower().endswith(('.png', '.jpg', '.jpeg'))],
                        # patch_sizes = patch_sizes, 
                        D = dico, cbpdn_params = cbpdn_params, device = device) # reload disabled means that we do not recode existing coeffs
    
# --- Analyzing Xs --- #
if do_analysis :
    imgp.make_psnr_sparseness_plots()
    
    dico = np.load(thin_path)['D']
    
    coeff_paths = ['./data/coeffs/'+x for x in os.listdir('./data/coeffs') if x.lower().endswith(('.npy')) and not 'psnrs' in x and not 'sparsenesses' in x]
    #coeff_paths = coeff_paths[:1000]
    
    # Computing the coeff activation for each image, a la SPORCO
    ca.histo_coeffs(coeff_paths=coeff_paths, dico = dico, patch_sizes = patch_sizes, 
                    bic = True, bicsize = .5) # reload disabled means we reload each coeff and redo the histogram
    
    # Todo here add the covariate 
    ca.ori_covariate(coeff_paths=coeff_paths,  dico = dico, patch_sizes = patch_sizes)
    
    # Computing the activation with respect to orientation 
    ca.ori_coeffs(coeff_paths=coeff_paths, dico = dico, patch_sizes = patch_sizes,
                N_theta = N_theta_thin, N_phase = N_phase_thin, N_Btheta = N_Btheta_thin, 
                bic = True, bicsize = .5) # reload disabled means we reload each coeff and redo the histogram

    ca.tsne_ori() 
    
    ca.make_delta_dirac(path = 'coeffs', do_norm = False,
                    N_theta = N_theta, N_Btheta = N_Btheta, N_phase = N_phase, bt_idx=0, do_bt=False,
                    do_plot=True, plot_color='gray')

# ---- CBPDN ---- #
#  Btheta x Theta #
# --------------- #
if do_CSC_btheta :
    # Running the CSC on images patches
    dico = np.load(full_path)['D']
    imgp.do_cbpdn_dataset(dataset = [data_resized+'/'+x for x in os.listdir(data_resized) if x.lower().endswith(('.png', '.jpg', '.jpeg'))],
                        patch_sizes = patch_sizes, D = dico, cbpdn_params = cbpdn_params, device = device, savepath = 'btheta') # reload disabled means that we do not recode existing coeffs
    
# --- Analyzing Xs --- #
if do_analysis_btheta :
    imgp.make_psnr_sparseness_plots(loadpath = 'btheta')
    
    dico = np.load(full_path)['D']
    
    coeff_paths = ['./data/btheta/'+x for x in os.listdir('./data/btheta') if x.lower().endswith(('.npy')) and not 'psnrs' in x and not 'sparsenesses' in x]
    #coeff_paths = coeff_paths[:1000]
    
    # Computing the coeff activation for each image, a la SPORCO
    ca.histo_coeffs(coeff_paths=coeff_paths, dico = dico, patch_sizes = patch_sizes, 
                    reload = False, bic = False, bicsize = .5, filepath = 'btheta') # reload disabled means we reload each coeff and redo the histogram
    
    # Todo here add the covariate 
    ca.ori_covariate(coeff_paths=coeff_paths,  dico = dico, patch_sizes = patch_sizes, 
                    reload = False, filepath = 'btheta')
    
    for chosen_bt in [0, N_Btheta//2, N_Btheta-1] :
        # Computing the activation with respect to orientation 
        ca.ori_coeffs(coeff_paths=coeff_paths, dico = dico, patch_sizes = patch_sizes,
                        N_theta = N_theta, N_phase = N_phase, N_Btheta = N_Btheta, 
                        reload = False, bic = False, bicsize = .5, filepath = 'btheta_%s' % chosen_bt,
                        do_btheta = True, chosen_bt = chosen_bt) # reload disabled means we reload each coeff and redo the histogram

    #ca.tsne_ori(filepath = 'btheta') 
    
    ca.make_delta_dirac(path = 'btheta', do_norm = False,
                    N_theta = N_theta, N_Btheta = N_Btheta, N_phase = N_phase, bt_idx=0, do_bt=True,
                    do_plot=True, plot_color='gray')
    

# CBPDN params
cbpdn_params = {'MaxMainIter' : 500, 'lmbda' : 1e-2, 'RelStopTol' : 1e-3, 'RelaxParam' : 1.8,
                'L1Weight' : 0.5, 'AutoRho_RsdlRatio' : 1.05, 'DataType' : np.float32} # need this just for the learning which is done on cpu w/ SPORCO

######################
# --- Part two--- #
######################    
if do_sporco_learning :
    # This is where we learn the dictionary using CBPDN (as used in the param scanning, i.e. with optimal parameters)
    if learning_bt :
        learning.learn_from_img(database_path = data_resized, D = np.load(full_path)['D'],
                            save_name = './data/dictionary_learned.npz',
                            n_epochs = 800, N_image_rec = 200, DEZOOM=1)
    if learning_thin :
        learning.learn_from_img(database_path = data_resized, D = np.load(thin_path)['D'],
                            save_name = './data/dictionary_learned_thin.npz',
                            n_epochs = 800, N_image_rec = 200, DEZOOM=1)
    if learning_scratch :
        learning.learn_from_img(database_path = data_resized, D = np.random.randn(filter_size, filter_size, N_theta*N_Btheta*N_phase),
                            save_name = './data/dictionary_12x12x108.npz',
                            n_epochs = 800, N_image_rec = 200, DEZOOM=1)
    
# delete the "DataType from cbpdn_params
del cbpdn_params['DataType']

if do_sporco_reconstructing :
    # This is where we reconstruct the images to make the PSNR/sparseness plots 
    # TODO maybe we need to do this on cpu
    if not os.path.isfile('./data/full/sparsenesses.npy') :
        reconstructing.reconstruct_from_img(database_path = data_resized, D = np.load(full_path)['D'],
                                            cbpdn_params=cbpdn_params, device = device, 
                                            savepath = 'full')
    if not os.path.isfile('./data/thin/sparsenesses.npy') :
        reconstructing.reconstruct_from_img(database_path = data_resized, D = np.load(thin_path)['D'],
                                            cbpdn_params=cbpdn_params, device = device, 
                                            savepath = 'thin')
        
    if not os.path.isfile('./data/learned/sparsenesses.npy') :
        reconstructing.reconstruct_from_img(database_path = data_resized, D = np.load('./data/dictionary_learned.npz')['D'],
                                            cbpdn_params=cbpdn_params, device = device, 
                                            savepath = 'learned')
    if not os.path.isfile('./data/learned_thin/sparsenesses.npy') :
        reconstructing.reconstruct_from_img(database_path = data_resized, D = np.load('./data/dictionary_learned_thin.npz')['D'],
                                            cbpdn_params=cbpdn_params, device = device, 
                                            savepath = 'learned_thin')
        
    if not os.path.isfile('./data/12x12x108/sparsenesses.npy') :
        reconstructing.reconstruct_from_img(database_path = data_resized, D = np.load('./data/dictionary_12x12x108.npz')['D'],
                                            cbpdn_params=cbpdn_params, device = device, 
                                            savepath = '12x12x108')

    # And here we redo the plots for Fig 2
    reconstructing.psnr_sparseness_plot()
    
    # And here the plot for post-learning effects (Fig 3)
    reconstructing.coeff_kde(init_dict = np.load(full_path)['D'],
                            learned_dict = np.load('./data/dictionary_learned.npz')['D'],
                            N_theta = N_theta, N_phase = N_phase, N_Btheta = N_Btheta,
                            figname = 'learned')
    
    reconstructing.coeff_kde(init_dict = np.load(thin_path)['D'],
                            learned_dict = np.load('./data/dictionary_learned_thin.npz')['D'],
                            N_theta = N_theta_thin, N_phase = N_phase_thin, N_Btheta = N_Btheta_thin,
                            figname = 'learned_thin')
    
    reconstructing.coeff_kde(init_dict = np.load(full_path)['D'],
                            learned_dict =  np.load(full_path)['D'],
                            N_theta = N_theta, N_phase = N_phase, N_Btheta = N_Btheta,
                            figname = 'full')
    
    # And we plot some reconstruction
    '''reconstructing.reconstruction_plot(D = np.load(full_path)['D'],
                                    database_path = data_all,
                                    coeff_path = 'full',
                                    images_paths = ['IMG_7103.jpg'],
                                    filename = 'example_bigsize',  cbpdn_params = cbpdn_params,
                                    N_X = patch_sizes[0], N_Y = patch_sizes[1], N_theta = N_theta, N_Btheta = N_Btheta, N_phase = N_phase,
                                    init_recompute = True)'''
        
    reconstructing.reconstruction_plot(D = np.load(full_path)['D'],
                                        database_path = data_resized,
                                        coeff_path = 'full',
                                        images_paths = ['IMG_0226.jpg', 'IMG_5998.jpg', 'IMG_7754.jpg'],
                                        filename = 'full',  cbpdn_params = cbpdn_params,
                                        N_X = patch_sizes[0], N_Y = patch_sizes[1], N_theta = N_theta, N_Btheta = N_Btheta, N_phase = N_phase)
    
    reconstructing.reconstruction_plot(D = np.load(thin_path)['D'],
                                        database_path = data_resized,
                                        coeff_path = 'thin',
                                        images_paths = ['IMG_0226.jpg', 'IMG_5998.jpg', 'IMG_7754.jpg'],
                                        filename = 'thin',  cbpdn_params = cbpdn_params,
                                        N_X = patch_sizes[0], N_Y = patch_sizes[1], N_theta = N_theta_thin, N_Btheta = N_Btheta_thin, N_phase = N_phase_thin
                                        )
    
    reconstructing.reconstruction_plot(D = np.load('./data/dictionary_learned.npz')['D'],
                                        database_path = data_resized,
                                        coeff_path = 'learned',
                                        images_paths = ['IMG_0226.jpg', 'IMG_5998.jpg', 'IMG_7754.jpg'],
                                        filename = 'learned',  cbpdn_params = cbpdn_params,
                                        N_X = patch_sizes[0], N_Y = patch_sizes[1], N_theta = N_theta, N_Btheta = N_Btheta, N_phase = N_phase)
    
    reconstructing.compute_img_differences(coeff_path_1 = 'full', coeff_path_2 = 'learned', coeffs = ['IMG_0226.jpg', 'IMG_5998.jpg', 'IMG_7754.jpg'])
    
if do_sporco_resilience :
    # Reconstruction with cutoff
    reconstructing.reconstruction_plot(D = np.load(full_path)['D'],
                                        database_path = data_resized,
                                        coeff_path = 'full',
                                        images_paths = ['IMG_0226.jpg'],
                                        filename = 'full',  cbpdn_params = cbpdn_params,
                                        N_X = patch_sizes[0], N_Y = patch_sizes[1], N_theta = N_theta, N_Btheta = N_Btheta, N_phase = N_phase,
                                        cutoffs = np.linspace(0.001, 0.5, 8))
    
    reconstructing.reconstruction_plot(D = np.load(thin_path)['D'],
                                        database_path = data_resized,
                                        coeff_path = 'thin',
                                        images_paths = ['IMG_0226.jpg'],
                                        filename = 'thin', cbpdn_params = cbpdn_params,
                                        N_X = patch_sizes[0], N_Y = patch_sizes[1], N_theta = N_theta_thin, N_Btheta = N_Btheta_thin, N_phase = N_phase_thin,
                                        cutoffs = np.linspace(0.001, 0.5, 8))
    
    reconstructing.reconstruction_plot(D = np.load('./data/dictionary_learned.npz')['D'],
                                        database_path = data_resized,
                                        coeff_path = 'learned',
                                        images_paths = ['IMG_0226.jpg'],
                                        filename = 'learned', cbpdn_params = cbpdn_params,
                                        N_X = patch_sizes[0], N_Y = patch_sizes[1], N_theta = N_theta, N_Btheta = N_Btheta, N_phase = N_phase,
                                        cutoffs = np.linspace(0.001, 0.5, 8))
    
    # This is where we do the cutoff
    n_cutoffs = 6
    archs = ['thin', 'full', 'learned', 'learned_thin', '12x12x108']
    archs = ['thin', 'full', 'learned', 'learned_thin']
    colormaps = [met_brewer.met_brew(name = "Degas", brew_type = "continuous", n = n_cutoffs)[::-1], 
                met_brewer.met_brew(name = "Derain", brew_type = "continuous", n = n_cutoffs)[::-1], 
                met_brewer.met_brew(name = "Archambault", brew_type = "continuous", n = n_cutoffs), 
                plt.cm.Oranges(np.linspace(0.2,0.9, n_cutoffs)), 
                plt.cm.Greys(np.linspace(0.9,0.2, n_cutoffs))] 
    colormaps = [met_brewer.met_brew(name = "Degas", brew_type = "continuous", n = n_cutoffs)[::-1], 
                met_brewer.met_brew(name = "Derain", brew_type = "continuous", n = n_cutoffs)[::-1], 
                met_brewer.met_brew(name = "Archambault", brew_type = "continuous", n = n_cutoffs), 
                plt.cm.Oranges(np.linspace(0.2,0.9, n_cutoffs))] 
    coeff_paths = archs
    dicos = [np.load(thin_path)['D'], np.load(full_path)['D'],
            np.load('./data/dictionary_learned.npz')['D'], np.load('./data/dictionary_learned_thin.npz')['D'],
            np.load('./data/dictionary_12x12x108.npz')['D']]
    dicos = [np.load(thin_path)['D'], np.load(full_path)['D'],
            np.load('./data/dictionary_learned.npz')['D'], np.load('./data/dictionary_learned_thin.npz')['D']]
                
    all_psnrs = np.zeros((len(archs), n_cutoffs, 200)) # this is to do stats afterwards
    all_sparsenesses = np.zeros_like(all_psnrs)
    for iarch, arch in enumerate(archs) :
        psnrs, sparsenesses = reconstructing.resilience_plot(dico = dicos[iarch], database_path = data_resized,
                                                            coeff_path = coeff_paths[iarch], filename=coeff_paths[iarch],
                                                            N_X = patch_sizes[0], N_Y = patch_sizes[1], N_theta = N_theta, N_Btheta = N_Btheta, N_phase = N_phase,
                                                            cutoffs = np.linspace(0.001, 0.5, all_psnrs.shape[1]),
                                                            colormap = colormaps[iarch], device = device,
                                                            lmbda = cbpdn_params['lmbda'], cbpdn_params = cbpdn_params, 
                                                            tot_images = all_psnrs.shape[-1])
        all_psnrs[iarch,:,:] = psnrs
        all_sparsenesses[iarch,:,:] = sparsenesses
        
    for i in range(4) :
        print(stats.mannwhitneyu(all_psnrs[1,i,:], all_psnrs[3,i,:])) # loggabor pre vs post learned
        print('-')
        print(stats.mannwhitneyu(all_psnrs[0,i,:], all_psnrs[1,i,:])) # loggabor thin vs loggabor 
        print('-')
        print(stats.mannwhitneyu(all_psnrs[4,i,:], all_psnrs[3,i,:], alternative = 'less')) # online thin vs online loggabor 
        print('-------------\n')


if do_sporco_param_scan :
    # And this is just the copy paste of the param scan code, which should work on its own just fine
    scan.run_scan(database_path = data_resized, D = np.load(full_path)['D'], reload_only = False)