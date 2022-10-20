import os, sys
from os.path import join, dirname, exists
from glob import glob
import h5py
import matplotlib.pyplot as plt 
import numpy as np
import random
import gc
from copy import copy, deepcopy

from joblib import Parallel, delayed
from pathlib import Path
from datetime import datetime
# sklearn
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, LeaveOneGroupOut, train_test_split
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# load configuration
import argparse
import importlib

# load functions from nitorch
sys.path.append(join(dirname(__file__), "../../nitorch/"))

import nitorch
from nitorch.transforms import *
from nitorch.trainer import Trainer
from nitorch.metrics import *
# from nitorch.utils import count_parameters
from nitorch.initialization import weights_init
from nitorch.data import *

sys.path.insert(1, dirname(__file__))
from CNNpipeline import cnn_pipeline
from models import *

torch.backends.cudnn.benchmark = True

OUTPUT_DIR = join(dirname(__file__),'results/')
##############################################################################################################

def main():
    
    parser = argparse.ArgumentParser('runCNNpipeline Setting')
    parser.add_argument('config', type=str, help='please add the configuration name')
    parser.add_argument('-d', '--debug', action='store_true', help='switch the debug mode')
    option = parser.parse_args()
    configname = option.config
    DEBUG_MODE = option.debug
    # load the config file
    pkg = importlib.import_module(f'config.{configname}')
    cfg = deepcopy(pkg.Config)
    cfg_path = os.path.abspath(sys.modules[cfg.__module__].__file__)
    
    # get the number of parallel processes to run assuming 1 per GPU
    PARALLEL_RUNS= len(cfg.GPUS)
    
    # In DEBUG mode use simpler settings
    if DEBUG_MODE:
        print("{} Running DEBUG MODE {}".format('='*20,'='*20))
        print(f"change RAND_STATE: {cfg.RAND_STATE} -> 42")
        cfg.RAND_STATE = 42
        if PARALLEL_RUNS>2:
            PARALLEL_RUNS = 2
            print(f"changing GPUs: {cfg.GPUS} -> {cfg.GPUS[:PARALLEL_RUNS]}")
            cfg.GPUS = cfg.GPUS[:PARALLEL_RUNS]
        if cfg.N_CV_TRIALS>2:
            print(f"changing N_CV_TRIALS: {cfg.N_CV_TRIALS} -> 2")
            cfg.N_CV_TRIALS = 2 
        print("Running all models with \n num_epochs: 5 \n batch_size: 4 \n earlystop_patience: 0")
        for lbl in cfg.ANALYSIS:
            for model_i in cfg.ANALYSIS[lbl]["MODEL_SETTINGS"]:
                model_i["num_epochs"]=5
                model_i["batch_size"]=4 
                model_i["earlystop_patience"]=0
                
    # set the random seed
    if cfg.RAND_STATE is not None:
        random.seed(cfg.RAND_STATE)
        np.random.seed(cfg.RAND_STATE)
        torch.manual_seed(cfg.RAND_STATE)
          
    # iterate over h5 files
    for h5_file in cfg.H5_FILES:
        # check if holdout split is provided as a separate h5
        if isinstance(h5_file, (list, tuple)): 
            h5_file, h5_hold_file = h5_file
        else:
            h5_hold_file = None

        start_time = datetime.now()
        # Create a folder to save the output and results
        outfoldername = start_time.strftime("%Y%m%d-%H%M")
        if cfg.OUT_FOLDER_SUFFIX: outfoldername = f"{cfg.OUT_FOLDER_SUFFIX}_{outfoldername}"
        SAVE_DIR = "{}/{}-{}/{}".format(
                OUTPUT_DIR, configname,
                os.path.basename(h5_file).replace('.h5','').replace('*',''),
                outfoldername)
        # if debug mode then save in the folder 'debug_run' instead
        if DEBUG_MODE: 
            # clear any previous debug mode results and outputs
            os.system(f"rm -rf {OUTPUT_DIR}/debug_run 2> /dev/null")
            SAVE_DIR = "{}/debug_run/{}".format(OUTPUT_DIR, outfoldername)
            
        if not os.path.isdir(SAVE_DIR): os.makedirs(SAVE_DIR)
        
        # Save the configuration file in the output folder
        if not len(os.listdir(SAVE_DIR))==0: 
            print(f"[WARN] output dir '{SAVE_DIR}' is not empty.. files might be overwriten..")
        cpy_cfg = f'{SAVE_DIR}/{configname}_{outfoldername}.py'
        os.system(f"cp {cfg_path} {cpy_cfg}")
            
        print(f"========================================================\
        \nRunning CNNpipeline on: {h5_file}\
        \nStart time:             {start_time}\
        \nSaving results at:      {SAVE_DIR}")

        # print GPU status
        devices = get_all_gpu_status(cfg.GPUS)

        # read the attributes in the data h5 files to create all [input -> output] pairs
        with h5py.File(glob(h5_file)[0], 'r') as data:

            assert 'X' in data.keys(), "the h5file {} must contain a column 'X' containing the imaging / input dataset.".format(h5_file)
            assert 'i' in data.keys(), "the h5file {} must contain a column 'i' containing unique IDs for each subject in the dataset.".format(h5_file)

            # generate all combinations of
            # (1) INPUT->OUTPUT combinations (2) MODEL pipeline (3) N_CV_TRIALS trials 
            # so that they can be run parallelly on GPUs
            settings = []
            
            for lbl, cfg_i in cfg.ANALYSIS.items():
                
                # check that all labels provided in ANALYSIS exist in the h5 file
                # make it case insensitive
                lbl = [col for col in data.keys() if lbl.lower()==col.lower()]
                assert len(lbl)!=0, f"Label '{lbl}' doesnt exist in h5file.\
 Existing columns in h5file: {list(data.keys())}"
                lbl = lbl[0]
                
                for model_setting_ in cfg_i["MODEL_SETTINGS"]: 
                    # to avoid overwriting the config 
                    model_setting = deepcopy(model_setting_) 
                    # intialize the model using the provided model settings
                    if 'model_name' in model_setting: print(f"Loaded model :: \
{model_setting['model_name']} for {{X}}->{{{lbl}}}\n-----------------------------------------")
                    model = model_setting.pop('model')(**model_setting["model_params"])   
                                                       
                    # pre-generate the test indicies for the outer CV as they need to run in parallel
                    if cfg.N_CV_TRIALS<=1:
                        splitter = ShuffleSplit(n_splits=cfg.N_CV_TRIALS, 
                                                random_state=cfg.RAND_STATE)
                    # if N_CV_TRIALS > 1 then do N-fold cross validation
                    else:  
                        splitter = StratifiedKFold(n_splits=cfg.N_CV_TRIALS, 
                                                   shuffle=True, 
                                                   random_state=cfg.RAND_STATE)
                    strat_by = data[lbl][:]
                    # for regression, bin lbl into 3 categories to stratify
                    if cfg_i["TASK_TYPE"] == 'regression':
                        strat_by = pd.cut(strat_by, bins=3, labels=False)
                        
                    for cv_idx, (train_idx, test_idx) in enumerate(splitter.split(strat_by, strat_by)):
                        settings.extend([{"data": h5_file, 
                                          "data_hold": h5_hold_file, 
                                          "data_val_idx": test_idx,                                 
                                          "X_name": 'X', "y_name": lbl, 
                                          "model": model, 
                                          "trial": cv_idx, 
                                          "task_type": cfg_i["TASK_TYPE"],
                                          "metrics":cfg_i["METRICS"],
                                          **model_setting, 
                                          }])
                        
        # print the different analysis settings that were prepared above
        print(f"running {len(settings)} different combinations of [DL models] x [inp -> out] x [CV trials]")
        for i, setting in enumerate(settings):
            print("({})\t {}\t {}-{}\t cv_i={}".format(
                            i, setting['model_name'], 
                            setting['X_name'], setting['y_name'],
                            setting['trial']))
        
        # runs all the experiment combinations in parallel and save the results in `run_{i}.csv`
        with Parallel(n_jobs=PARALLEL_RUNS) as parallel:
            parallel(
                delayed(
                        cnn_pipeline)(
                            **setting, 
                            gpu=devices[i%len(devices)],
                            output_dir=SAVE_DIR, save_model=True, 
                            save_figures=True, return_model=False,
                            run_id=f"run{i}", debug=DEBUG_MODE) 
                
                     for i, setting in enumerate(settings))
        
        del model
        # stitch together the csv results that were generated in parallel and save in a single csv file 
        df = pd.concat([pd.read_csv(csv) for csv in glob(f"{SAVE_DIR}/run*.csv")], ignore_index=True)      
        # delete the temp csv files generated in parallel
        os.system(f"rm {SAVE_DIR}/run*.csv")  
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop unnamed columns            
        df = df.sort_values(['model_name','inp','out','run_id']) # sort
        df.to_csv(f"{SAVE_DIR}/run.csv", index=False)               

        runtime=str(datetime.now()-start_time).split(".")[0]
        print("TOTAL RUNTIME: {} secs".format(runtime))
            
            
##############################################################################################################

def get_all_gpu_status(gpus):
    c = torch.cuda
    devices = [torch.device(f'cuda:{gpu}' if c.is_available() else 'cpu') for gpu in gpus]
    print(f"-------------------------- GPU  STATUS -----------------------------\
          \nTorch version:     {torch.__version__}\
          \nCUDA available?    {c.is_available()}\
          \ngraphic name:      {c.get_device_name()}\
          \nusing GPUs:        {[str(dev) for dev in devices]}\
          \nmemory alloc:      {[str(round(c.memory_allocated(dev)/1024**3,1))+' GB' for dev in devices]}\
          \nmemory cached:     {[str(round(c.memory_reserved(dev)/1024**3,1))+' GB' for dev in devices]}\
          \nrandom_seed:       {torch.initial_seed()}\
          \n--------------------------------------------------------------------\
          ")
    return devices

if __name__ == "__main__": main()