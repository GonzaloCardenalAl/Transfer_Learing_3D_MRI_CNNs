import os, sys
from glob import glob
import os.path as path

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

import h5py
import random 

# load helper functions from nitorch
from plotGraphs import show_brain


def plot_dists(h5, lbl, col_names, suptitle=''):
    
    df = pd.DataFrame({c:h5[c][()] for c in [lbl]+col_names})
    df.index = h5["i"][()]
    
    if len(df[lbl].unique())>2:
        print("lbl has more than 2 states and in range [{}, {}, {}] \n aborting distribution plots .....".format(
                df[lbl].min(), df[lbl].mean(), df[lbl].max())) 
    else:        
        f, axes = plt.subplots(1, len(col_names)+1, figsize=(4*(len(col_names)+1), 3))
        if suptitle: plt.suptitle(suptitle)

        val = dict(df[lbl].astype(int).astype('str').value_counts())
        axes[0].pie(list(val.values()), labels = list(val.keys()), autopct='%1.0f%%', shadow=True, startangle=90) 
        axes[0].axis('equal')

        for ax, c in zip(axes[1:], col_names):
            if len(df[c].unique())<=10: # then bar plot
                dfi = df[[c,lbl]].value_counts(dropna=False).unstack().sort_index()

                dfi.sort_index().plot.bar(
                    stacked=True, title=c, legend=True, 
                    alpha=0.5, sort_columns=True, 
                    grid=True, rot=90, ax=ax, width=0.9)
    #             for i, x in enumerate(dfi.values):
    #                 ax.text(i, x+(x/100)+1, str(x))

            else: # histogram
                for lbl_i, dfi in df.groupby(lbl):
                    lbl_i = int(lbl_i)
                    dfi[c].plot.hist(label="lbl="+str(lbl_i), title=c,
                                     grid=True, alpha=0.5, legend=True, bins=10, ax=ax)

        plt.legend()   
        plt.show()

def show_h5_contents(H5_FILES, ignore=''):
    
    if isinstance(H5_FILES, str): H5_FILES = [H5_FILES]
    
    for k, h5_files_glob in enumerate(H5_FILES):
        
        for h5_file in glob(h5_files_glob):
            print("______________________________________________________________________________________")
            if ignore and ignore in h5_file:
                print(f"ignoring '{h5_file}' because of the ignore str '{ignore}'")
            else:            
                with h5py.File(h5_file, 'r') as h5:
                    print("filename: {} \n h5.columns = {} \n h5.attrs = {}".format(
                        h5_file.split("/")[-1], list(h5.keys()), list(h5.attrs.keys())))

                    try:
                        n = len(h5["i"])
                        y = h5.attrs['labels'][0]
                        conf_names=h5.attrs["confs"].tolist()
                        print("n = {} \t label = {} \t X.shape = {}  \t confs = {}".format( 
                            n, y, h5["X"].shape, conf_names))

                        # plot the confs and label distributions in the data
                        print("label and conf with distributions:")
                        plot_dists(h5, y, conf_names)

                    except Exception as e:
                        print(e) 

                    # plot a random subject's brain image in the data 
                    random_i = random.randint(0, n-1)
                    img = h5["X"][random_i]
                    print("\nsubject: {} \t (min={:.2f}, mean={:.2f}, max={:.2f})".format(
                        h5["i"][random_i], img.min(), img.mean(), img.max()))

                    # in X contain MRI images then show the image
                    if len(img.shape) == 3:
                        # show the MRI
                        show_brain(img, cmap='hot')
                        plt.show()
                        # show voxel values 
                        fig = plt.figure(figsize=(3,3))
                        vals = img.ravel()
                        vals = vals[vals != 0]
                        plt.hist(vals, bins=20)
                        plt.title("Voxel values distribution (showing only non-zeros)")
                        plt.show()
                        print(f"Voxel value at [1,1,1]={img[1,1,1]} (expected val around the brain is 0)")