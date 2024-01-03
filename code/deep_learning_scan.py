'''
@hugoladret
THIS IS ABOUT SPARSE CODING AND THEN DOING DEEP LEARNING DIRECTLY ON THE COEFF, WHICH DOESNT SEEM TO PLAY NICELY WITH US
'''

import numpy as np
import matplotlib.pyplot as plt 
from tqdm import trange
import time 
import os

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


import torch_cbpdn as cbpdn
#import sparse_utils as utils
import deep_learning_dataset as dl_dataset
import deep_learning_train as dl_train

from sklearn.model_selection import ParameterGrid


if __name__ == '__main__' :
    #torch.multiprocessing.set_start_method('spawn')
    
    # ---------------------------------
    # PARAMS --------------------------
    # ---------------------------------
    do_csc = True 
    do_run = True 
    do_plot = True 
    use_amp = True

    # Below is for SC
    D = np.load('./data/dictionary_thin.npz')['D']
    

    # Sparse coding parameters 
    cbpdn_params = {'MaxMainIter' : 250, 'lmbda' : 0.01, 'RelStopTol' : 0.1, 'RelaxParam' : 1.8,
                    'L1Weight' : 0.5, 'AutoRho_RsdlRatio' : 1.05}

    # SUPER IMPORTANT SCALE PARAMETER
    nscale = 2 # upscales the image --> needs to be a power of 2 (usually 2 or 4 thus)

    # ceil list 
    ceil_list = [False, True, True]
    ceil_values = [0, 4 ,2]
    ceil_list = [False]
    ceil_values = [0]

    # Below is for the network
    # Random generator seed
    seed = 42

    # Meta parameters
    workers = 18 # batch loaders
    batch_size = 904 # batch size, increase to 1024 if possible to speed
    epochs = 100 # epochs
    
    # And amount of runs we'll be doing
    n_runs = 1 # number of runs
    
    arch = 'ResNet18' # one of VGG16, ResNet18, ResNet50

    param_grid = {'lrs': [0.001, 0.0001, 0.000001],
            'momentums': [0.9],
            'weight_decays': [5e-3, 5e-4, 5e-2],
            'optimizers' : ['SGD', 'AdamW']}
    
    param_grid = {'lrs': np.linspace(0.001, 0.00001, 5),
            'momentums': [0.9],
            'weight_decays': [5e-4, 5e-5],
            'optimizers' : ['Adam']}
    
    param_grid = {'lrs': np.linspace(0.0002, 0.0002, 1), # 0.001 est un peu trop haut, 0.0005 marche bien
            'momentums': [0.9],
            'weight_decays': [5e-2, 5e-1, 5e-3],
            'optimizers' : ['Adam']}
    
    

    # ---------------------------------
    # RUNNING SPARSE CODING -----------
    # ---------------------------------
    if do_csc :
        if torch.cuda.is_available() :
            device = torch.device('cuda') 
        else :
            device = torch.device('cpu')
            
        # Load dictionary
        D = torch.tensor(D, dtype = torch.float32, device = device)

        dl_dataset.run_encoding(D = D, nscale = nscale, cbpdn_params = cbpdn_params,
                                device = device, batch_size = batch_size, workers = workers)

        torch.cuda.empty_cache()



    # ---------------------------------
    # RUNNING DL ----------------------
    # ---------------------------------
    if do_run :
        def run_dl(arch, optim,
                ceil_list, ceil_values, 
                workers, batch_size, epochs, 
                lr, momentum, weight_decay, 
                n_runs) :
            for irun in range(n_runs) :
                for i, _ in enumerate(ceil_list) :
                    filenames_suffix = 'exploseptembre_%s_optim%s_lr%.5f_epochs%d_mom%.5f_decay%.4f' %(arch, optim, lr, epochs, momentum, weight_decay)
                    do_ceil = ceil_list[i] 
                    ceil_val = ceil_values[i] 
                    print('> About to start running a network with the following filename :')
                    print(filenames_suffix)
                    
                    if arch == 'VGG16' :
                        model = dl_train.VGG('VGG16', D.shape[-1])
                    elif arch == 'ResNet18' :
                        model = dl_train.ResNet18()
                    elif arch == 'ResNet50' :
                        model = dl_train.Resnet50()
                        
                    model = model.to(device, memory_format=torch.channels_last)
                    model = torch.nn.DataParallel(model)
                    
                    cudnn.benchmark = True
                    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
                    
                    criterion = nn.CrossEntropyLoss()
                    criterion.cuda()
                    
                    if optim == 'SGD' :
                        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
                    elif optim == 'AdamW' :
                        optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
                    elif optim == 'Adam' :
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
                    np.save('./model/%s%s_accs_run_%s.npy' % (filenames_suffix, ceil_val, irun), val_accs)
                    np.save('./model/%s%s_losses_run_%s.npy' % (filenames_suffix, ceil_val, irun), val_losses)

                    
                    fig, axs = plt.subplots(figsize = (15,5), ncols = 2)
                    axs[0].plot(val_accs)
                    axs[0].set_title('Max validation accuracy: %.3f' % np.max(val_accs))
                    axs[1].plot(val_losses)
                    axs[1].set_title('Min validation loss: %.3f' % np.min(val_losses))
                    for ax in axs :
                        ax.set_xlim(0, epochs-1)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                    fig.savefig('./model/%s%s_val_accs_losses_run_%s.png' % (filenames_suffix, ceil_val, irun))
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
                    fig.savefig('./model/%s%s_train_accs_losses_run_%s.png' % (filenames_suffix, ceil_val, irun))
                    plt.close(fig)
                    
                    del model, optimizer, train_loader, val_loader, train_accs, train_losses, val_accs, val_losses
                    del transform_test, transform_train, cifar_train_loader, cifar_val_loader
                    del criterion
                    torch.cuda.empty_cache()
            

        for iparam, param in enumerate(ParameterGrid(param_grid)) :
            print('Running a DNN with grid # %s/%s' % (iparam, len(ParameterGrid(param_grid))))
            run_dl(arch = arch, optim = param['optimizers'],
                ceil_list = ceil_list, ceil_values = ceil_values, 
                workers = workers, batch_size = batch_size, epochs = epochs, 
                lr = param['lrs'], momentum = param['momentums'], weight_decay = param['weight_decays'], 
                n_runs = n_runs)
            print('\n')

    # ---------------------------------
    # PLOTTING DL ----------------------
    # ---------------------------------
    if do_plot :
        # Initialize variables to store the sum of losses and accuracies across runs
        sum_val_losses = None
        sum_val_accs = None
        sum_train_losses = None
        sum_train_accs = None

        # Loop through each run and each ceil_value to load the saved data
        for ceil_val in ceil_values:
            for irun in range(n_runs):
                val_losses = np.load('./model/%s%s_losses_run_%s.npy' % (filenames_suffix, ceil_val, irun))
                val_accs = np.load('./model/%s%s_accs_run_%s.npy' % (filenames_suffix, ceil_val, irun))
                # Assume train losses and accs are also saved in a similar manner
                train_losses = np.load('./model/%s%s_train_losses_run_%s.npy' % (filenames_suffix, ceil_val, irun))
                train_accs = np.load('./model/%s%s_train_accs_run_%s.npy' % (filenames_suffix, ceil_val, irun))

                if sum_val_losses is None:
                    sum_val_losses = np.zeros_like(val_losses)
                    sum_val_accs = np.zeros_like(val_accs)
                    sum_train_losses = np.zeros_like(train_losses)
                    sum_train_accs = np.zeros_like(train_accs)

                sum_val_losses += val_losses
                sum_val_accs += val_accs
                sum_train_losses += train_losses
                sum_train_accs += train_accs

            # Compute averages
            avg_val_losses = sum_val_losses / n_runs
            avg_val_accs = sum_val_accs / n_runs
            avg_train_losses = sum_train_losses / n_runs
            avg_train_accs = sum_train_accs / n_runs

            # Plotting
            fig, axs = plt.subplots(figsize=(15, 5), ncols=2)

            axs[0].plot(avg_val_accs, label=f'Ceil Value: {ceil_val}')
            axs[0].set_title('Average Validation Accuracy Across Runs : max %.4f +- %.4f' % (np.max(avg_val_accs), np.std(avg_val_accs)))
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Accuracy')

            axs[1].plot(avg_val_losses, label=f'Ceil Value: {ceil_val}')
            axs[1].set_title('Average Validation Loss Across Runs')
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Loss')

            for ax in axs:
                ax.legend()
                ax.set_xlim(0, len(avg_val_accs) - 1)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

            plt.savefig(f'./model/avg_val_accs_losses_ceil_{ceil_val}.png')
            plt.close(fig)

            fig, axs = plt.subplots(figsize=(15, 5), ncols=2)

            axs[0].plot(avg_train_accs, label=f'Ceil Value: {ceil_val}')
            axs[0].set_title('Average Train Accuracy Across Runs')
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Accuracy')

            axs[1].plot(avg_train_losses, label=f'Ceil Value: {ceil_val}')
            axs[1].set_title('Average Train Loss Across Runs')
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Loss')

            for ax in axs:
                ax.legend()
                ax.set_xlim(0, len(avg_train_accs) - 1)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

            plt.savefig(f'./model/avg_train_accs_losses_ceil_{ceil_val}.png')
            plt.close(fig)

