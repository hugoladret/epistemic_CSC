'''
@hugoladret
> Run the sparse encoding directly saving the coeffs (X) on the HDD
> Usage : The script will take care of running the sparse coding, saving the results, and running VGG on them afterwards, with defined threshold (so multiple runs)
> DO REMOVE BOTH FOLDERS IN ./data/cifar_sparse BETWEEN RUNS AS THEY TAKE ABOUT 200GO on an SSD
'''

import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import time 
import os
import torch.nn.functional as F

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

import torch_cbpdn as cbpdn
#import sparse_utils as utils
import deep_learning_dataset as dl_dataset

import warnings
warnings.filterwarnings("ignore") # complex32 warning otherwise


# ---------------------------------
# METHODS VGG----------------------
# ---------------------------------
import torch
import torch.nn as nn
class VGG(nn.Module):
    def __init__(self, vgg_name, in_channels):
        cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }   
        
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], in_channels)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class ModifiedVGG(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ModifiedVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




# ---------------------------------
# METHODS RESNET-------------------
# ---------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                            planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(144, 64, kernel_size=3,
                            stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout2d(p=0.5)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])




# ---------------------------------
# METHODS DATASET-------------------
# ---------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, img_labels, img_dir, device, transform,
                do_ceil, ceil, 
                do_rgb, N_theta, N_Btheta):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.device = device
        self.do_ceil = do_ceil 
        self.ceil = ceil 
        self.do_rgb = do_rgb
        self.N_theta = N_theta 
        self.N_Btheta = N_Btheta

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = np.load(self.img_dir + '' + str(idx) + '.npy')
        
        if self.do_rgb :
            image = image.reshape((image.shape[0], image.shape[1], self.N_theta, self.N_Btheta , 2 ))
            image = image.sum(axis = -1)
            image = self.sparse_code_to_rgb(image, self.N_Btheta, self.N_theta,)
            
        #image = image.reshape((image.shape[0], image.shape[1], self.N_theta, self.N_Btheta , 2 ))
        #image = image.sum(axis = -1)
        #image = image.reshape((image.shape[0], image.shape[1], -1))
        
        image = torch.as_tensor(image, dtype=torch.float32)
        image = image.swapaxes(0, -1)
        
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)  # Apply transforms
            
        if self.do_ceil:
            valceil = image.max() / self.ceil
            idxs_thresh = torch.where((image < valceil) & (image > -valceil))
            image[idxs_thresh] = 0
        
        return image, label
    
    def sparse_code_to_rgb(self, coeffs, N_Btheta=1, N_theta=144):
        im_RGB = np.zeros((coeffs.shape[0], coeffs.shape[1], 3))
        
        # Precompute thetas for the loop
        thetas = np.linspace(0, 180, N_theta, endpoint=False)

        for i_bt in range(N_Btheta):
            for i_theta, theta_ in enumerate(thetas):
                im_abs = 1. * np.flipud(np.fliplr(np.abs(coeffs[:, :, i_theta, i_bt])))
                RGB = np.array([.5*np.sin(2*theta_ + 2*it*np.pi/3)+.5 for it in range(3)])
                # Create RGB array similar to the original code
                #RGB = np.array([0.5 * np.sin(2 * theta_ + 2 * it * np.pi / 3) + 0.5 for it in range(3)])
                
                im_RGB += im_abs[:, :, np.newaxis] * RGB[np.newaxis, np.newaxis, :]

        im_RGB /= im_RGB.max()
        im_RGB = np.flip(im_RGB, axis = (0,1))
        plt.imshow(im_RGB)
        plt.show()
        return im_RGB


    

    
def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())

def run_one_epoch(loader, model, criterion, optimizer, epoch, device,
                runmode = 'train', use_amp = False, scaler = None, scheduler = None) :
    
    if runmode == 'train' :
        model.train() 
    elif runmode == 'val' :
        model.eval() 
    else :
        raise 
    
    running_loss = 0.
    running_accuracy = 0.
    for i, (inp, target) in enumerate(loader):

        # Send data to GPU
        if torch.cuda.is_available(): 
            inp, target = inp.cuda(),target.cuda()
            inp = inp.to(memory_format=torch.channels_last)
        
        if runmode == 'val' :
            with torch.no_grad():
                # Compute model output
                output = model(inp)
                loss = criterion(output, target)
        else :
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                # Compute model output
                output = model(inp)
                loss = criterion(output, target)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # compute gradient and do SGD step
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            #loss.backward()
            #optimizer.step()
            
        running_loss += loss.item()
        running_accuracy += accuracy(output, target)
        
    running_loss /= len(loader)
    running_accuracy /= len(loader)
    
    return running_accuracy, running_loss
