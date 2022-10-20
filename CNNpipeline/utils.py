import os, sys
from os.path import join, dirname
from glob import glob
import matplotlib.pyplot as plt 
import numpy as np

import gc, time

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# load functions from nitorch
sys.path.append(join(dirname(__file__), "../../nitorch/"))
from nitorch.data import show_brain
from nitorch.utils import *


#######################  helper functions   ########################

# delete all tensors present on the current GPU and free up space
def clear_reset_gpu(gpu, print_debug=False):
    try:
        for obj in gc.get_objects():
            if torch.is_tensor(obj) or (hasattr(obj,'data') and
               torch.is_tensor(obj.data) and obj.get_device()==int(str(gpu).replace('cuda:',''))):
                if print_debug: print(f"deleting ... cuda{obj.get_device()}: ", type(obj), obj.size())                     
                del obj
    except: pass
    time.sleep(5)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(20)
    
    
def show_MRI_stats(X_i, title=None, save_fig_path=''):
    show_brain(X_i, cmap='hot')
    plt.suptitle(title)
    if save_fig_path: 
        plt.savefig(save_fig_path+'.jpg', 
                    bbox_inches='tight', pad_inches=0.01, dpi=50)
        plt.close()
    else:
        plt.show()
    # plot voxel value distribution
    vals = X_i.ravel()
    vals = vals[vals != 0]
    plt.figure(figsize=(3,3))
    plt.hist(vals, bins=20)
    plt.title("voxel values distribution (showing only non-zeros)")
    plt.xlabel(f"Voxel value at [1,1,1] = {X_i[1,1,1]} (expected val 0)")
    plt.tight_layout()
    if save_fig_path: 
        plt.savefig(save_fig_path +'_dist.jpg', 
                    bbox_inches='tight', pad_inches=0.01, dpi=50)
        plt.close()
    else:
        plt.show()