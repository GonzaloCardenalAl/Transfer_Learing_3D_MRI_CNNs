"""
Author: Roshan Rane 
Date: Feb 2022 
This dataset Class is combined with Sudeshna Bora's lab rotation code. """
# Global imports
import os, sys
import glob
from os.path import join

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snszoom

import h5py
from slugify import slugify
from tqdm.notebook import tqdm

from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

import nibabel as nib

import json
import re 

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)) + "/../../helper/")
from plotGraphs import plotGraph


class UKBB2020:

    def __init__(self,
                 BIDS_DIR = "/ritter/share/data/UKBB_2020/BIDS",
                 META_DATA = "/ritter/share/data/UKBB_2020/tables/sourceData_40682.csv",
                 META_DATA_DICT="/ritter/share/data/UKBB_2020/tables/Data_Dictionary_Showcase.tsv",
                 META_DATA_DICT_CODING="/ritter/share/data/UKBB_2020/tables/Codings.tsv"):

        self.BIDS_DIR = BIDS_DIR
        assert os.path.isfile(self.BIDS_DIR+"/participants.csv"), f"BIDS_DIR {self.BIDS_DIR} is not a valid BIDS dir"
        self.META_DATA = META_DATA
        assert os.path.isfile(self.META_DATA), f"META_DATA file {self.META_DATA} doesn't exist"
        
        self.META_DATA_DICT = pd.read_csv(META_DATA_DICT, encoding="ISO-8859-1", sep='\t', 
                                          usecols=['FieldID','Field','ValueType', "Coding"])
        
        self.META_DATA_DICT_CODING = pd.read_csv(META_DATA_DICT_CODING, encoding="ISO-8859-1", sep='\t',
                                                 index_col=["Coding", "Value"], usecols=["Coding","Value", "Meaning"])

        #load the csv containing all meta data
        self.df = pd.read_csv(BIDS_DIR+"/participants.csv", index_col='subjectID')
            
        print(f"total subjects in df: {len(self.df)}")             
            
        # init split var to 'all'
        self.split = 'all'
        # init variables needed for h5file creations
        self.df_h5 = pd.DataFrame(index=self.df.index)
        self.hdf5_name_x = ""
        self.X_colnames = []
        self.all_labels = []
        self.all_confs = []
        self.lbl_suffix = {}
        self.conf_suffix = {}
        
    
    ### META DATA COLUMNS    
    def get_metadata(self, predefined=["demographics"], cols=[],
                        rename_cols=True, print_cols=True, split='all'):
        """
        split: allowed values ['train', 'holdout', 'all']
        """
        split = split.lower()
        assert split in ['train', 'holdout', 'all']
        # separate the holdout set if requested
        if split == 'train':
            valid_eids = self.df[~self.df["holdout"]].index.tolist()
        elif split == 'holdout':
            valid_eids = self.df[self.df["holdout"]].index.tolist()
        else: # split == 'all'
            valid_eids = self.df.index.tolist()
            
        # load valid colnames            
        with open(os.path.dirname(os.path.abspath(__file__))+"/columns.json", "r") as fp:
            valid_cols = json.load(fp)
            
        with open(os.path.dirname(os.path.abspath(__file__))+"/column_queries.json", "r") as fp:
            predefined_qrs = json.load(fp)
            
        predefined = [qr.lower() for qr in predefined]
        invalid_qrs = [qr for qr in predefined if qr not in predefined_qrs.keys()]
        if invalid_qrs:
            print(f"[WARN]: skipping {invalid_qrs} as they are not valid predefined query in `column_queries.json`. \nCurrently supported queries are {list(predefined_qrs.keys())}")  
            for qr in invalid_qrs: predefined.remove(qr)
        
        # load the 'predefined' query to column names from `column_queries.json`        
        usecols = []
        for qr in predefined:
            if qr not in invalid_qrs:
                usecols.extend(predefined_qrs[qr])                 
        
        # if a regex is given, get all columns from META_DATA_DICT that match this regex
        if cols:
            newcols = []
            for col in cols:
                if '*' in col:
                    reg = re.compile(col)
                    matched_cols = [valid_col for valid_col in valid_cols if reg.match(valid_col)]
                    if not matched_cols:
                        print(f"[WARN]: skipping col regex {col} as it didnt match anything")
                else:
                    if col in valid_cols:
                        matched_cols = [col]
                    else:
                        print(f"[WARN]: skipping col {col} as it is invalid")
                
                newcols.extend(matched_cols)
            
            usecols.extend(newcols)
            
        # ensure once that all cols are valid_columns
        usecols = [c for c in usecols if c in valid_cols]
        
        # for all usecols query the resp. description, category names mapped, and determine the resp. dtypes
        col_descs, col_cat_remap, col_dtypes, col_dates = self._get_metadata_desc(usecols)
#         print('col_dtypes', col_dtypes)
#         print('col_cat_remap',col_cat_remap)
        if print_cols: print('loaded following columns:\n', col_descs)
              
        # add subjectID
        usecols=['eid']+usecols
        col_descs.update({'eid':'subjectID'})
        col_dtypes.update({'eid':'int'})
        
        with open(os.path.dirname(os.path.abspath(__file__))+'/eids_to_rows.json', "r") as fp:
            eids_to_rows = json.load(fp)
        skiprows= [int(i)+1 for eid, i in eids_to_rows.items() if int(eid) not in valid_eids] # +1 is used to skip the header
        if len(skiprows):
            print("skipping {}/{} subjects not belonging to {} split".format(len(skiprows), len(self.df), split))
            
        # read the requested meta data cols and return the df
        df = pd.read_csv(self.META_DATA, 
                         usecols=usecols, dtype=col_dtypes, 
                         skiprows=skiprows, header=0, 
                         parse_dates=col_dates, infer_datetime_format=True,
                         encoding="ISO-8859-1")
        
        if rename_cols:
            # remap values in categorical dtypes
            for col, cat_remap in col_cat_remap.items():
                df[col] = df[col].cat.rename_categories(cat_remap)
                # change nan values to a 'missing' category
                df[col] = df[col].cat.add_categories("missing").fillna("missing")

            # rename columns from the code to description names 
            df = df.rename(columns=col_descs).set_index('subjectID') 
        else:
            df = df.set_index('eid')
                
        return df
    
    def _get_metadata_desc(self, cols):
        col_descs = {}
        col_cat_remap = {}
        col_dtypes = {}   
        col_dates = []    
        dtype_map = {'Integer':'float', 'Continuous':'float',
                     'Categorical single':"category", 'Categorical multiple':"category", 
                     'Date':'object', 'Time':'object', 
                     'Text':'object',  'Compound':'object'}
        
        for col in cols:            
            fieldID, ses = col.split('-')
            col_info = self.META_DATA_DICT[self.META_DATA_DICT.FieldID==int(fieldID)]
            assert len(col_info)==1, f"Field {col} matched several rows ({len(col_info)}) in META_DATA_DICT file"
            # get description from META_DATA_DICT 
            desc = col_info.Field.values[0]
            # append the assessment session info in the title if it is not 2.0
            if ses != '2.0':  desc += f" (s{ses})"
            col_descs.update({col: desc})
            # get dtype from META_DATA_DICT    
            col_type = col_info.ValueType.values[0]
            col_dtypes.update({col: dtype_map[col_type]})
            if col_type in ['Date','Time']: col_dates.extend([col])
            # get values encoding of categorical values from META_DATA_DICT_CODING
            if 'Categorical' in col_type:
                remap = self.META_DATA_DICT_CODING.loc[col_info.Coding.values[0]].squeeze()
                try:
                    remap.index = remap.reset_index()['Value'].apply(lambda x: str(float(int(x))))
                except Exception as e: pass #print(e)
                col_cat_remap.update({col: remap.to_dict()})

        return col_descs, col_cat_remap, col_dtypes, col_dates
        
    def plot_metadata(self, df, plot_types=[]):
        
        # if plot_types are not explicitly provided then infer them from dtypes
        if not plot_types:
            new_plt_types = []
            plot_types_map = {"float64": "hist",
                              "binary": "pie",
                              "category": "barh",
                              "int64": "hist",
                              "datetime64[ns]": "hist",
                              "object": "bar"}
            for col in df:
                dtype =  df[col].dtype.name
                dtype = 'binary' if (dtype in ['category', 'float64', 'int64', 'bool'] and len(df[col].unique())<=4) else dtype
                new_plt_types.extend([plot_types_map[dtype]])
#                 print(dtype, col)        
            plot_types = new_plt_types
        
        n=len(df.columns)
        nrows = n//3+int((n%3)>0)
        ncols = min(3,n)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
        if n>1: 
            axes = axes.ravel()
        else:
            axes = [axes]
        
        if len(plot_types)==1 and n!=1:
            plot_types = plot_types*n
            
        for i, col in enumerate(df):    
#             print(col, plot_types[i])
            # if name is too long add a newline
            title = col
            if len(col)>30:
                words = col.split(' ')
                title = ' '.join(col.split(' ')[:len(words)//2])+'\n'+' '.join(col.split(' ')[len(words)//2:])
            plotGraph(df, col, plt_type=plot_types[i], ax=axes[i], title=title)

        # remove any extra axes
        if len(axes)>n: [ax.axis("off") for ax in axes[n:]]
            
        plt.tight_layout()
        
        
    ### PREPARE INPUT (X) and OUTPUT (y) IN df_h5 FOR WRITING TO H5 ###
    def add_var_to_h5(self, dfq, col, y_colname=None, typ='lbl',
                   binarize=False, class0=None, class1=None,
                   norm=False, viz=False):
        """
        Loads a label or conf from the given dataframe 'dfq' into a dataframe 
        that will exported as a h5 file 'self.df_h5'
        dfq:  is a dataframe with the subject 'eid's as index
        """    
        typ = typ.lower()
        assert typ in ['lbl', 'conf']
        
        # if no different name is specified for the new column, then use the current col name itself
        if not y_colname: y_colname = col.lower()
        self.df_h5.loc[:,y_colname] = dfq.loc[:,col]
        # convert categorical and object dtypes to bool/int type
        if self.df_h5[y_colname].dtype.name in ['category', 'object']:
            if self.df_h5[y_colname].dtype.name == 'object': 
                self.df_h5[y_colname] = pd.Categorical(self.df_h5[y_colname])
            codes = self.df_h5[y_colname].cat.codes.replace(-1,np.nan).astype(float)
            self.df_h5[y_colname] = codes
        
        # Convert the label/conf to binary if requested
        bin_suffix = ''
        
        if binarize:
            assert (class0 is not None) and (class1 is not None),\
"if 'binarize' is requested then class0 and class1 cases should be specified."
        
            q = self.df_h5[y_colname].copy()
            self.df_h5[y_colname] = np.nan # reset values to add binary values
            
            if isinstance(class0, int):
                self.df_h5.loc[(q<=class0), y_colname] = 0 
                bin_suffix += "l{}".format(class0)
            elif isinstance(class0, list):
                self.df_h5.loc[q.isin(class0), y_colname] = 0 
                bin_suffix += "l{}".format("".join([str(y) for y in class0]))
            else:
                print("[ERROR] binarizing failed. Incoherent values given for class0=", class0) 
                                           
            if isinstance(class1, int):
                self.df_h5.loc[q>=class1, y_colname] = 1
                bin_suffix += "u{}".format(class1)
            elif isinstance(class1, list):
                self.df_h5.loc[q.isin(class1), y_colname] = 1
                bin_suffix += "u{}".format("".join([str(y) for y in class1]))
            else:
                print("[ERROR] binarizing failed. Incoherent values given for class1=", class1) 
            
            
        #  normalize to range [0,1] if requested
        if norm==True and not binarize:
            self.df_h5[y_colname] = (self.df_h5[y_colname]-self.df_h5[y_colname].min())/(self.df_h5[y_colname].max()-self.df_h5[y_colname].min())
        
        if viz:
                if len(self.df_h5[y_colname].unique())<=3:
                    plotGraph(self.df_h5, y_colname, plt_type='bar')
                    map_xlabels={'0.0':'Class 0', '1.0':'Class 1', 'nan':'dropped'}
                    plt.gca().set_xticklabels([map_xlabels[t.get_text()] for t in plt.gca().get_xticklabels()])                    
                else:
                     plotGraph(self.df_h5, y_colname, plt_type='hist')
                plt.show()                
        
        if typ=='lbl':
            if y_colname not in self.all_labels: self.all_labels.extend([y_colname])     
            # save the labelname in file name (also append binarizing rules to the final filename)
            if bin_suffix: self.lbl_suffix.update({y_colname: bin_suffix})   
        elif typ=='conf':
            if y_colname not in self.all_confs: self.all_confs.extend([y_colname])
            if bin_suffix: self.conf_suffix.update({y_colname: bin_suffix})   
            
        return self.df_h5
        
        
    def prepare_X(self, 
                  preloaded_X=None, mri_col="",
                  viz=True):

        assert self.df_h5 is not None, "first assign a label from the questionnaires for the hdf5 conversion.\nUse the methods self.load_label() and self.binarize_label()"
        
        # if a preloaded MRI data is provided directly as a dict of {subjectID: loaded_data_array}
        if preloaded_X is not None and (isinstance(preloaded_X, (pd.DataFrame, pd.Series))):
            if 'eid' == preloaded_X.index.name:
                preloaded_X.index.name = 'subjectID'
            elif 'eid' in preloaded_X.columns:
                preloaded_X = preloaded_X.rename(columns={'eid':'subjectID'})
            assert 'subjectID' in [preloaded_X.index.name]+list(preloaded_X.columns) , "\
the provided 'preloaded_X' must have one column (or index) called 'subjectID' containing the subject IDs."
            
            self.X_colnames = list(preloaded_X.columns)
            self.df_h5 = self.df_h5.merge(preloaded_X, on="subjectID")      
            # name of output hdf5
            self.hdf5_name_x = "IDPs-"
        # if a MRI column name from BIDS csv file is provided then set this path as X 
        elif mri_col: 
             # name of output hdf5 - path_T1_brain_to_MNI
            self.hdf5_name_x = "{}".format(
                mri_col.replace("path_","").replace("_","").replace("-","")) 
            self.X_colnames = [self.hdf5_name_x]
            self.df_h5[self.hdf5_name_x] = self.df[mri_col]
        else:        
            assert "Either provide IDPs in the 'preloaded_X' arg, or provide the MRI image column name \
 from BIDS participants.csv in the 'mri_col' arg. But neither of the two have been provided."

        # only retain rows that have both X data and label data available
        self.df_h5 = self.df_h5.dropna(subset=self.all_labels+self.X_colnames)
        print("n={} after dropping subjects with NaN".format(len(self.df_h5)))
        
    
# ### SAVE TO HDF5 ###
    def save_h5(self, filename_prefix='', mri_kwargs={"z_factor":1.}):
#                     drop_useless_voxels=False
        # if mri downsample is requested, add this info in the final hdf5 filename
        if "z_factor" in mri_kwargs:
            z_info = "z{:1.0f}".format(1/mri_kwargs["z_factor"])
            if mri_kwargs["z_factor"] != 1 and z_info not in self.hdf5_name_x:
                self.hdf5_name_x += z_info
            
        # add all lbls and confs present, to the h5file name
        if self.all_labels:
            hdf5_name_y = 'l'
            for lbl in self.all_labels:
                lbl += self.lbl_suffix[lbl] if (lbl in self.lbl_suffix) else ''
                hdf5_name_y += '-'+lbl
        if self.all_confs:
            hdf5_name_y += '-c'
            for conf in self.all_confs:
                conf += self.conf_suffix[conf] if (conf in self.conf_suffix)  else ''
                hdf5_name_y += '-'+conf
        
        filename = "{}-{}-n{}".format(self.hdf5_name_x, 
                                      hdf5_name_y,
                                      len(self.df_h5))
        
        if filename_prefix: filename = filename_prefix + '_' + filename 
        
        dest = join(self.BIDS_DIR, "../h5files", slugify(filename)+".h5")      
        # if file already exists, then print error and exit
        if os.path.isfile(dest): 
            print(f"hdf5 file already exists at {dest}. First delete it manually.")
            return
        else:
            print(f"saving h5 file at {dest}") 
            
        X = self.df_h5[self.X_colnames].to_numpy()
        # if MRI paths are provided as 'X' then load these MRI images into a np array
        if isinstance(X[0,0], str):
            mri_paths = list(self.df_h5[self.X_colnames[0]])
            X = self._load_mri(mri_paths, **mri_kwargs)
        else: # isinstance (X[0,0],(np.ndarray, np.generic)):
            X = np.stack(X)
            
        print(f"Writing into h5 file:")
        
        with h5py.File(dest, "w") as h5:
            print(f'writing X of shape {X.shape}..')
            h5.create_dataset('X', data=X, chunks=True) #
            h5.create_dataset('i', data=self.df_h5.index)
            print(f'Writing {len(self.all_labels)} labels and {len(self.all_confs)} confounds..')
            for y in self.all_labels:
                h5.create_dataset(y, data=self.df_h5[y])
            for c in self.all_confs:
                h5.create_dataset(c, data=self.df_h5[c])
            #set attributes to distinguish labels from confounds
            print(f'Setting attributes')
            h5.attrs['labels']= self.all_labels
            h5.attrs['confs']= self.all_confs
            #set attribute to name the X features for later interpretations
            h5.attrs['X_col_names']= self.X_colnames

        print(f"Finished writing h5.")
                
    
    def _load_mri(self, paths, 
                  z_factor=1., z_order=3, z_prefilter=True ,
                  apply_mask='/ritter/share/misc/masks/FSL_atlases/MNI/standard/MNI152_T1_1mm_brain_mask.nii.gz'):

        print("Extracting {} images into a single matrix..".format(len(paths)))  
        data = []
        
        for i, path in tqdm(enumerate(paths), total=len(paths)):
            img_path = join(self.BIDS_DIR, path)
            img_arr = nib.load(img_path).get_fdata().astype(np.float32)
            
            if apply_mask:
                mask_arr = nib.load(apply_mask).get_fdata()
                
            # interpolate (zoom) to a smaller size and remove decimals created from the interpolation
            if z_factor != 1:
                img_arr = zoom(img_arr, z_factor, 
                               order=z_order, prefilter=z_prefilter) 
                if apply_mask:
                    mask_arr = zoom(mask_arr, z_factor, order=0) # order=0 -> nearest neighbor interpolation
                # if not masking out the artifacts created in empty regions by zoom(order>1) operation,
                # atleast round off to int values as a hack
                elif z_order>1 and np.mean(img_arr)>1:
                    img_arr = np.around(img_arr, 0)
#                 plt.imshow(img_arr[:, :, 40])

            if apply_mask: 
                img_arr[mask_arr==0] = 0     
                
            data.extend([img_arr])
            
        data_matrix = np.stack(data)

        print("Finished loading MRIs into a data matrix of {:.2f} mb".format(sys.getsizeof(data_matrix)/1000000))
        
        return data_matrix