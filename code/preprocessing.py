import os
from PIL import Image 
from tqdm import tqdm 
import itertools

import matplotlib.pyplot as plt
from met_brewer import met_brew

import numpy as np 
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from LogGabor import LogGabor
from sporco import util

# Resizes the images to 1920x1080 for easier processing
def resize_images(paths, target_size) :
    # For each image in the paths variable, resize them to target_size 
    newpaths = []
    for path in tqdm(paths, total = len(paths), desc = 'Resizing . . .') :
        img = Image.open(path)
        w,h = img.size 
        square_side = min(w,h) # should always be h, but you never know 
        
        # Crop the image to a square
        left = (w - square_side) // 2
        top = (h - square_side) // 2
        right = (w + square_side) // 2
        bottom = (h + square_side) // 2
        
        img_cropped = img.crop((left, top, right, bottom))
        
        img_cropped = img_cropped.resize(target_size)
        #img = img.convert('L')
        
        os.makedirs('imgs/resized', exist_ok=True)
        newpath = './imgs/resized/' + 'IMG' + path.split('IMG')[1]
        img_cropped.save(newpath)
        newpaths.append(newpath)
        
    return newpaths

# Label the images' semantic content using resnet50
def label_images(paths) :
    # GPT4 written
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained model and move it to the GPU if available
    model = models.resnet50(ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.eval()
    
    preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()

    labels = {}
    # Open the file and parse each line
    with open('./data/imagenet_classes.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split(':')
            labels[int(key)] = value.strip()  # Convert the key to an integer

    # Prepare an empty dictionary for frequency counting
    frequency_dict = {}

    # Loop through all images
    for ipath, path in tqdm(enumerate(paths), desc = 'Labeling images . . .', total = len(paths)) :
        # Load the image
        img = Image.open(path)
        
        # Preprocess the image and move it to the GPU if available
        img_t = preprocess(img).to(device)
        batch_t = torch.unsqueeze(img_t, 0)
        
        # Predict the labels
        with torch.no_grad():
            out = model(batch_t)
        _, predicted = torch.max(out, 1)
        predicted_label = labels[predicted.item()]
        
        # Add the labels to the dictionary
        if predicted_label in frequency_dict:
            frequency_dict[predicted_label] += 1
        else:
            frequency_dict[predicted_label] = 1

    # Sort the dictionary by values in descending order and create a new dictionary
    sorted_frequency_dict = {k: v for k, v in sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True)}

    print('10 most frequent labels:')
    for key, value in itertools.islice(sorted_frequency_dict.items(), 10):
        print(f'{key}: {value}')
        
    # Select the first 10 items
    first_items = dict(itertools.islice(sorted_frequency_dict.items(), 15))

    # Get labels and their corresponding frequencies
    labels = list(first_items.keys())
    labels = [x.split(',')[0] for x in labels] # strip of the alias, do mention this in the paper
    frequencies = np.asarray(list(first_items.values())) / len(paths)

    # Create a bar chart
    fig, ax = plt.subplots(figsize = (8,6))
    
    barcolors = met_brew(name = 'Hokusai1', n = len(frequencies), brew_type='continuous')[::-1]
    ax.bar(np.linspace(0, 1, len(frequencies)), frequencies, width = 1/len(frequencies),
            edgecolor = 'w', color = barcolors)
    
    ax.set_xticks(np.linspace(0, 1, len(frequencies)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlim(0 - 1/len(frequencies), 1.0 + 1/len(frequencies))
    
    ax.set_ylim(0, 1.1 * max(frequencies))
    ax.set_yticks(np.linspace(0, 1.1 * max(frequencies), 5))
    ax.set_ylabel('Frequency')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig('./figs/image_labels.pdf', bbox_inches='tight')


# Generates the dictionaries for sparse coding
def generate_dicos(filter_size,
                N_theta_thin, N_Btheta_thin, N_phase_thin, thin_path,
                N_theta, N_Btheta, N_phase, full_path) :
    parameterfile = 'https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py'
    lg = LogGabor(parameterfile)
    filter_size = 12 # size of each dico's element --> sqrt(LG size)
    lg.set_size((filter_size, filter_size))

        
    thetas = np.linspace(0, np.pi, N_theta, endpoint = False)
    phases = np.linspace(0, np.pi, N_phase, endpoint=False)
    B_thetas = np.linspace(0, np.pi/6, N_Btheta+1, endpoint = True)[1:]
    K = N_theta * N_Btheta * N_phase
    
    i_ = 0
    D = np.zeros((filter_size, filter_size, K))
    for i_theta in range(N_theta):
        for i_Btheta in range(N_Btheta):
            for i_phase in range(N_phase):
                #print(i_)
                params= {'sf_0':.4, 'B_sf': lg.pe.B_sf, 'theta':thetas[i_theta], 'B_theta': B_thetas[i_Btheta]}
                env = lg.loggabor(filter_size // 2, filter_size // 2, **params) * np.exp(-1j * phases[i_phase])
                D[:, :, i_] = lg.normalize(lg.invert(env)*lg.mask)
                i_ += 1
    np.savez_compressed(full_path, D=D)
    del i_, D
    
    thetas_thin = np.linspace(0, np.pi, N_theta_thin, endpoint = False)
    B_thetas_thin = np.linspace(np.max(B_thetas), 0, N_Btheta_thin, endpoint = False) / 2.5
    phases_thin = np.linspace(0, np.pi, N_phase_thin, endpoint=False)
    K_thin = N_theta_thin * N_Btheta_thin * N_phase_thin
    
    i_ = 0
    D = np.zeros((filter_size, filter_size, K_thin))
    for i_theta in range(N_theta_thin):
        for i_Btheta in range(N_Btheta_thin):
            for i_phase in range(N_phase_thin):
                #print(i_)
                params= {'sf_0':.4, 'B_sf': lg.pe.B_sf, 'theta':thetas_thin[i_theta], 'B_theta': B_thetas_thin[i_Btheta]}
                env = lg.loggabor(filter_size // 2, filter_size // 2, **params) * np.exp(-1j * phases_thin[i_phase])
                D[:, :, i_] = lg.normalize(lg.invert(env)*lg.mask)
                i_ += 1
    np.savez_compressed(thin_path, D=D)
    
    # We also need to get the 12x12x108 from SPORCO
    D = util.convdicts()['G:12x12x108']
    np.savez_compressed('./data/dictionary_12x12x108.npz', D = D)
    
    print('Dictionaries generated, K_thin = %s, K = %s' % (K_thin, K))
    
    