'''
@hugoladret
THIS IS ABOUT SPARSE CODING AND THEN DOING DEEP LEARNING DIRECTLY ON THE COEFF, WHICH DOESNT SEEM TO PLAY NICELY WITH US
'''

import numpy as np
import matplotlib.pyplot as plt 
from tqdm import trange
import time 
import os
import shutil

import warnings
warnings.filterwarnings("ignore") # complex32 warning otherwise

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import OneCycleLR
import met_brewer

import torch_cbpdn as cbpdn
#import sparse_utils as utils
import deep_learning_dataset as dl_dataset
import deep_learning_train as dl_train

from sklearn.model_selection import ParameterGrid


if __name__ == '__main__' :
    #torch.multiprocessing.set_start_method('spawn')
    
    dicos = ['./data/dictionary_thin.npz',
            './data/dictionary_learned.npz', './data/dictionary_learned_thin.npz',
            './data/dictionary.npz']
    types = ['THIN', 'LEARNED', 'LEARNED_THIN', 'MB_BROAD']
    
    archs = ['thin', 'full', 'learned', 'learned_thin']
    cmaps = [met_brewer.met_brew(name = "Degas", brew_type = "continuous", n = 3)[::-1], 
            met_brewer.met_brew(name = "Archambault", brew_type = "continuous", n = 3), 
            plt.cm.Oranges(np.linspace(0.2,0.9, 3)), 
            met_brewer.met_brew(name = "Derain", brew_type = "continuous", n = 3)[::-1]] 
    
    for itype, csc_type in enumerate(types) :
        print('RUNNING FOR CSC TYPE %s' % csc_type)
        # ---------------------------------
        # PARAMS --------------------------
        # ---------------------------------
        do_csc = False 
        do_run = False 
        do_plot = False 
        do_summary_plot = True
        use_amp = True

        # Below is for SC
        D = np.load(dicos[itype])['D']
        
        # Sparse coding parameters 
        cbpdn_params = {'MaxMainIter' : 250, 'lmbda' : 0.01, 'RelStopTol' : 0.1, 'RelaxParam' : 1.8,
                        'L1Weight' : 0.5, 'AutoRho_RsdlRatio' : 1.05}

        # SUPER IMPORTANT SCALE PARAMETERrr
        nscale = 2 # upscales the image --> needs to be a power of 2 (usually 2 or 4 thus)

        # ceil list 
        ceil_list = [False, True, True]
        ceil_values = [0, 4 ,2]

        # Below is for the network
        # Random generator seed
        seed = 42

        # Meta parameters
        workers = 18 # batch loaders
        batch_size = 800 # batch size, increase to 1024 if possible to speed
        #batch_size = 960
        epochs = 100 # epochs
        
        # And amount of runs we'll be doing
        n_runs = 4 # number of runs
        
        momentum = 0.9 
        weight_decay = 5e-2 
        lr = 0.0002

        #csc_type = 'BROAD'
        filenames_suffix = 'runseptember_%s_%s_optim%s_lr%.5f_epochs%d_mom%.5f_decay%.4f' %(csc_type, 'Resnet18', 'Adam', lr, epochs, momentum, weight_decay)
        
        cmap = cmaps[itype]
        
        if torch.cuda.is_available() :
            device = torch.device('cuda') 
        else :
            device = torch.device('cpu')
                
        # ---------------------------------
        # RUNNING SPARSE CODING -----------
        # ---------------------------------
        if do_csc :            
            # Load dictionary
            D = torch.tensor(D, dtype = torch.float32, device = device)

            dl_dataset.run_encoding(D = D, nscale = nscale, cbpdn_params = cbpdn_params,
                                    device = device, batch_size = batch_size, workers = workers)

            torch.cuda.empty_cache()

        # ---------------------------------
        # RUNNING DL ----------------------
        # ---------------------------------
        if do_run :
            for irun in range(0, n_runs) : # THIS GUY
                print('>>> RUN # %d out of %d (starting at 0)' % (irun, n_runs-1))
                torch.manual_seed(seed + irun)
                np.random.seed(seed+irun)
                for i, _ in enumerate(ceil_list) :
                    
                    do_ceil = ceil_list[i] 
                    ceil_val = ceil_values[i] 
                    print('>> RUN for ceil value %d' % ceil_val)
                    print('> About to start running a network with the following filename :')
                    print(filenames_suffix)
                    
                    model = dl_train.ResNet18()
                        
                    model = model.to(device, memory_format=torch.channels_last)
                    model = torch.nn.DataParallel(model)
                    
                    cudnn.benchmark = True
                    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
                    
                    criterion = nn.CrossEntropyLoss()
                    criterion.cuda()

                    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
                        
                    transform_train = transforms.Compose([
                        #transforms.RandomCrop(32, 4),
                        transforms.RandomHorizontalFlip(),
                        transforms.Resize(size = (32,32)), # c'était bien la peine de faire tout en 196 pixels
                        transforms.Normalize(mean=[0.5],std=[0.225]),
                    ])
                    transform_test = transforms.Compose([
                        transforms.Resize(size = (32,32)),
                        transforms.Normalize(mean=[0.5],std=[0.225]),
                    ])

                    cifar_train_loader = torch.utils.data.DataLoader(
                        datasets.CIFAR10(root='./data', train=True, transform= transform_train),
                        batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
                    cifar_val_loader = torch.utils.data.DataLoader(
                        datasets.CIFAR10(root='./data', train=False, transform=transform_test),
                        batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)

                    train_loader = torch.utils.data.DataLoader(
                                        dl_train.CustomImageDataset(img_labels = cifar_train_loader.dataset.targets,
                                                                    img_dir = './data/cifar_sparse/X_train/',
                                                                    transform = transform_train, 
                                                                    device = device, do_ceil = do_ceil, 
                                                                    ceil = ceil_val,
                                                                    do_rgb=False, N_Btheta=1, N_theta = 72),
                    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
                    val_loader = torch.utils.data.DataLoader(
                                        dl_train.CustomImageDataset(img_labels = cifar_val_loader.dataset.targets,
                                                                    img_dir = './data/cifar_sparse/X_val/',
                                                                    transform = transform_test, 
                                                                    device = device, do_ceil = do_ceil, 
                                                                    ceil = ceil_val, 
                                                                    do_rgb=False, N_Btheta=1, N_theta = 72),
                        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

                    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
                    
                    train_accs, train_losses = [], []
                    val_accs, val_losses = [], []
                    print('All set, firing up DL training !')
                    tbar = trange(epochs, desc='Training', leave=True)
                    prev_val = 0
                    for epoch in tbar:
                        # train for one epoch
                        train_acc, train_loss = dl_train.run_one_epoch(train_loader, model, criterion, optimizer, epoch, device = device, runmode = 'train',
                                                                    use_amp = use_amp, scaler = scaler, scheduler = scheduler)

                        # evaluate on validation set
                        val_acc, val_loss = dl_train.run_one_epoch(val_loader, model, criterion, optimizer, epoch, device = device, runmode = 'val',
                                                                use_amp = use_amp, scaler = scaler, scheduler = scheduler)
                        
                        
                        train_accs.append(train_acc.cpu().numpy())
                        train_losses.append(train_loss)
                        val_accs.append(val_acc.cpu().numpy())
                        val_losses.append(val_loss)
                        
                        tqdm_desc = f"Training - latest acc : {val_acc.cpu().numpy():.4f} - delta acc : {val_acc.cpu().numpy() - prev_val:.4f}"
                        prev_val = val_acc.cpu().numpy()
                        tbar.set_description(tqdm_desc, refresh = True)
                        
        
                    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': np.max(train_accs)},
                                        os.path.join('./model/', '%s%s_checkpoint_%s_run_%s.tar' % (filenames_suffix, ceil_val, epoch, irun)))
                    np.save('./model/%s_ceil%s_accs_run_%s.npy' % (filenames_suffix, ceil_val, irun), val_accs)
                    np.save('./model/%s_ceil%s_losses_run_%s.npy' % (filenames_suffix, ceil_val, irun), val_losses)
                    np.save('./model/%s_ceil%s_train_accs_run_%s.npy' % (filenames_suffix, ceil_val, irun), train_accs)
                    np.save('./model/%s_ceil%s_train_losses_run_%s.npy' % (filenames_suffix, ceil_val, irun), train_losses)

                    
                    fig, axs = plt.subplots(figsize = (15,5), ncols = 2)
                    axs[0].plot(val_accs)
                    axs[0].set_title('Max validation accuracy: %.3f' % np.max(val_accs))
                    axs[1].plot(val_losses)
                    axs[1].set_title('Min validation loss: %.3f' % np.min(val_losses))
                    for ax in axs :
                        ax.set_xlim(0, epochs-1)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                    fig.savefig('./model/%s_ceil%s_val_accs_losses_run_%s.png' % (filenames_suffix, ceil_val, irun))
                    plt.close(fig)    
                    
                    fig, axs = plt.subplots(figsize = (15,5), ncols = 2)
                    axs[0].plot(train_accs)
                    axs[0].set_title('Max train accuracy: %.3f' % np.max(train_accs))
                    axs[1].plot(train_losses)
                    axs[1].set_title('Min train loss: %.3f' % np.min(train_losses))
                    for ax in axs :
                        ax.set_xlim(0, epochs-1)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                    fig.savefig('./model/%s_ceil%s_train_accs_losses_run_%s.png' % (filenames_suffix, ceil_val, irun))
                    plt.close(fig)
                    
                    del model, optimizer, train_loader, val_loader, train_accs, train_losses, val_accs, val_losses
                    del transform_test, transform_train, cifar_train_loader, cifar_val_loader
                    del criterion
                    torch.cuda.empty_cache()


        # ---------------------------------
        # PLOTTING DL ----------------------
        # ---------------------------------
        if do_plot :
            print('Plotting time ! Reloading filenames with prefix : %s' % filenames_suffix)
            # Initialize variables to store the sum of losses and accuracies across runs
            sum_val_losses = None
            sum_val_accs = None
            sum_train_losses = None
            sum_train_accs = None

            # Loop through each run and each ceil_value to load the saved data
            for ceil_val in ceil_values:
                all_val_losses = []
                all_val_accs = []
                all_train_losses = []
                all_train_accs = []
                for irun in range(n_runs):
                    val_losses = np.load('./model/%s_ceil%s_losses_run_%s.npy' % (filenames_suffix, ceil_val, irun))
                    val_accs = np.load('./model/%s_ceil%s_accs_run_%s.npy' % (filenames_suffix, ceil_val, irun))
                    # Assume train losses and accs are also saved in a similar manner
                    train_losses = np.load('./model/%s_ceil%s_train_losses_run_%s.npy' % (filenames_suffix, ceil_val, irun))
                    train_accs = np.load('./model/%s_ceil%s_train_accs_run_%s.npy' % (filenames_suffix, ceil_val, irun))

                    all_val_losses.append(val_losses)
                    all_val_accs.append(val_accs)
                    all_train_losses.append(train_losses)
                    all_train_accs.append(train_accs)

                avg_val_losses = np.asarray(all_val_losses).mean(axis = 0)
                avg_val_accs = np.asarray(all_val_accs).mean(axis = 0)
                avg_train_losses = np.asarray(all_train_losses).mean(axis = 0)
                avg_train_accs = np.asarray(all_train_accs).mean(axis = 0)
                
                std_val_losses= np.asarray(all_val_losses).std(axis = 0)
                std_val_accs = np.asarray(all_val_accs).std(axis = 0)
                std_train_losses = np.asarray(all_train_losses).std(axis = 0)
                std_train_accs = np.asarray(all_train_accs).std(axis = 0)

                # Plotting
                fig, axs = plt.subplots(figsize=(15, 5), ncols=2)

                axs[0].plot(avg_val_accs, label=f'Ceil Value: {ceil_val}')
                axs[0].fill_between(np.arange(len(avg_val_accs)), avg_val_accs - std_val_accs,
                                    avg_val_accs + std_val_accs, alpha=0.5)
                axs[0].set_title('Average Validation Accuracy Across Runs : max %.4f +- %.4f' % (np.max(avg_val_accs), std_val_accs[np.argmax(avg_val_accs)]))
                axs[0].set_xlabel('Epochs')
                axs[0].set_ylabel('Accuracy')

                axs[1].plot(avg_val_losses, label=f'Ceil Value: {ceil_val}')
                axs[1].fill_between(np.arange(len(avg_val_losses)), avg_val_losses - std_val_losses,
                                    avg_val_losses + std_val_losses, alpha=0.5)
                axs[1].set_title('Average Validation Loss Across Runs')
                axs[1].set_xlabel('Epochs')
                axs[1].set_ylabel('Loss')

                for ax in axs:
                    #ax.legend()
                    ax.set_xlim(0, len(avg_val_accs) - 1)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                axs[0].set_ylim(0, 1.0)
                
                plt.tight_layout()
                plt.savefig(f'./model/final/{csc_type}_avg_val_accs_losses_ceil_{ceil_val}.pdf', format = 'pdf', bbox_inches = 'tight', dpi = 200)
                plt.close(fig)


                fig, axs = plt.subplots(figsize=(15, 5), ncols=2)

                axs[0].plot(avg_train_accs, label=f'Ceil Value: {ceil_val}')
                axs[0].fill_between(np.arange(len(avg_train_accs)), avg_train_accs - std_train_accs,
                                    avg_train_accs + std_train_accs, alpha=0.5)
                axs[0].set_title('Average Validation Accuracy Across Runs : max %.4f +- %.4f' % (np.max(avg_train_accs), std_train_accs[np.argmax(avg_train_accs)]))
                axs[0].set_title('Average Train Accuracy Across Runs')
                axs[0].set_xlabel('Epochs')
                axs[0].set_ylabel('Accuracy')

                axs[1].plot(avg_train_losses, label=f'Ceil Value: {ceil_val}')
                axs[1].fill_between(np.arange(len(avg_train_losses)), avg_train_losses - std_train_losses,
                                    avg_train_losses + std_train_losses, alpha=0.5)
                axs[1].set_title('Average Train Loss Across Runs')
                axs[1].set_xlabel('Epochs')
                axs[1].set_ylabel('Loss')

                for ax in axs:
                    #ax.legend()
                    ax.set_xlim(0, len(avg_train_accs) - 1)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                axs[0].set_ylim(0, 1.0)

                plt.tight_layout()
                plt.savefig(f'./model/final/{csc_type}_avg_train_accs_losses_ceil_{ceil_val}.pdf', format = 'pdf', bbox_inches = 'tight', dpi = 200)
                plt.close(fig)
                
                
        if do_summary_plot :
            print('Summary plotting time ! Reloading filenames with prefix : %s' % filenames_suffix)

            # Loop through each run and each ceil_value to load the saved data
            fig, axs = plt.subplots(figsize=(15, 5), ncols=2)
            
            for iceil, ceil_val in enumerate(ceil_values):
                all_val_losses = []
                all_val_accs = []
                all_train_losses = []
                all_train_accs = []
                for irun in range(n_runs):
                    val_losses = np.load('./model/%s_ceil%s_losses_run_%s.npy' % (filenames_suffix, ceil_val, irun))
                    val_accs = np.load('./model/%s_ceil%s_accs_run_%s.npy' % (filenames_suffix, ceil_val, irun))
                    # Assume train losses and accs are also saved in a similar manner
                    train_losses = np.load('./model/%s_ceil%s_train_losses_run_%s.npy' % (filenames_suffix, ceil_val, irun))
                    train_accs = np.load('./model/%s_ceil%s_train_accs_run_%s.npy' % (filenames_suffix, ceil_val, irun))

                    all_val_losses.append(val_losses)
                    all_val_accs.append(val_accs)
                    all_train_losses.append(train_losses)
                    all_train_accs.append(train_accs)

                avg_val_losses = np.asarray(all_val_losses).mean(axis = 0)
                avg_val_accs = np.asarray(all_val_accs).mean(axis = 0)
                avg_train_losses = np.asarray(all_train_losses).mean(axis = 0)
                avg_train_accs = np.asarray(all_train_accs).mean(axis = 0)
                
                std_val_losses= np.asarray(all_val_losses).std(axis = 0)
                std_val_accs = np.asarray(all_val_accs).std(axis = 0)
                std_train_losses = np.asarray(all_train_losses).std(axis = 0)
                std_train_accs = np.asarray(all_train_accs).std(axis = 0)
                

                axs[0].plot(avg_val_accs, label=f'Ceil Value: {ceil_val}', color= cmap[iceil])
                axs[0].fill_between(np.arange(len(avg_val_accs)), avg_val_accs - std_val_accs,
                                    avg_val_accs + std_val_accs, alpha=0.5, color= cmap[iceil])
                axs[0].set_xlabel('Epochs')
                axs[0].set_ylabel('Accuracy')

                axs[1].plot(avg_val_losses, label=f'Ceil Value: {ceil_val}', color= cmap[iceil])
                axs[1].fill_between(np.arange(len(avg_val_losses)), avg_val_losses - std_val_losses,
                                    avg_val_losses + std_val_losses, alpha=0.5, color= cmap[iceil])
                axs[1].set_xlabel('Epochs')
                axs[1].set_ylabel('Loss')

            for ax in axs:
                #ax.legend()
                ax.set_xlim(0, len(avg_val_accs) - 1)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
            axs[0].set_ylim(0, 1.0)
            
            plt.tight_layout()
            plt.savefig(f'./model/final/summary_plot_{csc_type}.pdf', format = 'pdf', bbox_inches = 'tight', dpi = 200)
            plt.close(fig)



        # delete the folder ./data/cifar_sparse/X_train
        '''print('Deleting sparse coeffs')
        shutil.rmtree('./data/cifar_sparse/X_train')
        shutil.rmtree('./data/cifar_sparse/X_val')
        print('All done for this dico !! \n\n\n')
        '''