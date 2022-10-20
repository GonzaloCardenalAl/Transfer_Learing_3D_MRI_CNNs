# dependencies and plotting
import os, sys
from os.path import join, dirname
from contextlib import redirect_stdout
from glob import glob
import h5py
import matplotlib.pyplot as plt 
import numpy as np
import copy, io
# path
from pathlib import Path
from datetime import datetime
# from joblib.externals.loky.backend.context import get_context
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# import torch.multiprocessing as mp
# mp.set_start_method('fork')
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# load functions from nitorch
sys.path.append(join(dirname(__file__), "../../nitorch/"))
from nitorch.transforms import  * 
from nitorch.callbacks import EarlyStopping, ModelCheckpoint
from nitorch.trainer import Trainer
from nitorch.metrics import *
from nitorch.utils import *
# from nitorch.utils import count_parameters
from nitorch.initialization import weights_init
from nitorch.data import *

sys.path.append(dirname(__file__))
from utils import *
from dataloaders import *

def cnn_pipeline(
    model, gpu,
    data, data_val_idx=None, data_hold='',
    batch_size=16, num_epochs=100,
    X_name='X', y_name='y',
    criterion = nn.CrossEntropyLoss, criterion_params ={'reduction':'mean'},
    balance_loss_weights=False, 
    optimizer = optim.Adam, optimizer_params = {"lr":1e-4, "weight_decay":1e-4},
    scheduler = optim.lr_scheduler.StepLR, scheduler_params={"step_size":30, 'gamma':0.1},
    rescale_X = True, rescale_Y = False, augmentations = [], 
    earlystop_patience=15, save_model='best',
    metrics=[balanced_accuracy_score],
    model_weights_init=True, show_grad_flow=False, 
    output_dir="results/", run_id='run0', save_figures=False,
    return_model=True, debug=False,
    task_type='classif',
    **kwargs
): 
    """
    Parameters
    ----------
    model: An intialized pytorch model (torch.nn.Module)
    gpu: the GPU index to when running the CNNpipeline
    
    data           : a h5 file or a dict containing  {'X': np.array(*MRIdata*), 'y': np.array(*label*)}
    data_hold      : the holdout/test data provided in same format as 'data'
                     on which the the trained models will be reevaluated.
    X_name, y_name : defines which keys to use as input 'X' and as the label 'y'
                      in the provided data h5 files / dicts
    data_val_idx: the sample indexes in 'data' that will be used as validation data.
                  (default) If set to None, then the 20% of 'data' is randomly sampled as validation.
    batch_size: batch_size. If it is too high and the GPU throws out-of-memory error then this function
                will automatically reduce it by 2 and try again.
    num_epochs: maximum number of epochs to train the model
    criterion, criterion_params: the pytorch loss function along with the arguments (dict) to pass to it.
                                 (https://pytorch.org/docs/stable/nn.html#loss-functions) 
    balance_loss_weights      : If set to True, calculates balancing weights for the different classes that 
                                helps to counter the effect of unbalanced classes in the data.
                                 As an example, refer 'weights' parameter in torch.nn.CrossEntropyLoss.
    optimizer, optimizer_params: the pytorch training optimizer algorithm along with the arguments (dict) 
                                 to pass to it including the learning rate 'lr'
                                 (https://pytorch.org/docs/stable/optim.html#algorithms) 
    scheduler, scheduler_params: the pytorch training optimizer algorithm along with the arguments (dict) 
                                 to pass to it including the learning rate 'lr'
                                 (https://pytorch.org/docs/stable/optim.html#algorithms) 
    model_weights_init : If set to True (default) randomly initializes all trainable parameters
                         Set this to False when using pre-trained models or other custom initialized models
                         
    rescale_X          : If True, rescales the voxels values to [0,1] using nitorch's 
                         IntensityRescale augmentation feature.
    augmentations      : A list of augmentations to apply on training data (refer nitorch/transforms.py)
    earlystop_patience : If set to zero, no early stopping is performed. Otherwise will 
                         use EarlyStopping functionality  (refer nitorch/callbacks.py) and 
                         sets patience to provided number of epochs.
    save_model         : If set as 'best', the model is saved at its 'best' performance, in the output_dir
                         If set as 'checkpoints' then models are saved whenever there is an improvement
                         If set as False or empty str then no models are saved.
    metrics            : A list of metric functions to use to evaluate the model performance.
    task_type          : can be one of ['classif', 'classif_bin', 'regression', 'other']
    output_dir   : the output directory in which the training log, saved model, 
                   and finals results (run_xx.csv) are stored
    run_id       : a unique prefix to use when saving all outputs generated such as
                   training log, model checkpoints, plots and finals results file
    save_figures : saves all intermediary plots generated during training in the output_dir
    Returns
    ----------
    model: trained model
    results: training and testing results in a pandas Series
    """
    
    # write all outputs to a separate .log file
    with open(join(output_dir, run_id)+".log", "w") as logfile:
        with redirect_stdout(logfile):
            
            start_time = datetime.now()
            print("START_TIME \t {}".format(start_time.strftime("%H:%M::%S")), flush=True)
            
            # STEP 1) LOAD DATA
            # if h5file is provided then load it 
            if isinstance(data, str) and os.path.exists(data):
                data = h5py.File(data, 'r')
                
            if not debug:
                X_data = data[X_name][:]
                y_data  = data[y_name][:]
                i_data  = data['i'][:]
                # if validation data indices are not manually provided then randomly sample them
                if data_val_idx is None:
                    data_val_idx = np.random.randint(0, len(y_data), 
                                                     size=round(0.2*len(y_data)))
                    print("Since 'data_val_idx' is not explicitly provided, \
randomly sampled n={}/{} as validation data".format(len(data_val_idx), len(y_data)))

            # In DEBUG mode, use only a small subset of the data stratified
            else:
                debug_size=50
                X_data = data[X_name][:debug_size]
                y_data  = data[y_name][:debug_size]
                i_data  = data['i'][:debug_size]
                # override train and val idxs
                train_idx, data_val_idx = np.arange(0,debug_size-10), np.arange(debug_size-10,debug_size)
                print(f"[DEBUG] using only {debug_size} data points")
                
            data_load_time = int((datetime.now() - start_time).total_seconds())
            print("dataset loaded successfully in {}m:{}s..".format(data_load_time//60, data_load_time%60), flush=True)
            
            
        
            #Normalizing Y between 0,1 values
            if rescale_Y == True:
                y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))
                
       
            try:
                data.close()
            except:
                pass
            del data
            
            # set the provided GPU as the cuda device
            with torch.cuda.device(gpu):
                
                start_time = datetime.now()
                FINISHED = False
                
                params_log = {
                    "run_id"           : run_id,
                    "model"            : model.__class__.__name__,
                    "model_params"     : kwargs['model_params'] if 'model_params' in kwargs else 'NA',
                    "model_unique_name": kwargs['model_name'] if 'model_name' in kwargs else 'NA',
                    "inp"              : X_name,
                    "out"              : y_name,
                    "n_samples"        : len(y_data),
                    "m__batch_size":  batch_size,
                    "m__num_epochs":  num_epochs,
                    "m__criterion":   f"{criterion.__name__}({criterion_params})",
                    "m__optimizer":   f"{optimizer.__name__}({optimizer_params})",
                    "m__scheduler":   f"{scheduler.__name__}({scheduler_params})",
                    "m__augmentations": [type(aug).__name__ for aug in augmentations],
                    "m__earlystop_patience": earlystop_patience}

                print(f"Starting CNN_PIPELINE() with:\
                \n{params_log}\
                \noutput_dir: {output_dir}\
                \nusing GPU:  {str(gpu)} \
                \n---------------- CNNpipeline starting ----------------", flush=True)
                
                while(not FINISHED):
                    print_debug = False
                    save_fig_prefix = f"{output_dir}/{run_id}_" 
                    # prepare a dict to store all results
                    result = params_log 
                        # store additional info passed by runCNNpipeline for results.csv in the results directly
                    result.update(kwargs)
                    
                    try:
                        # initialize DL model on the GPU
                        model = model.cuda() 
                        # if there are unknown keys in kwargs then raise a warning at least
                        for key,val in kwargs.items(): 
                            if key not in ['trial','model_params','model_name']:
                                print(f"[WARN] Unknown arg CNNpipeline({key}={val}).\
Is it unintentional / spelling mistake ?")
                        result.update({"m__description": str(model)}) # save model full structure as str
                        m__n_params = count_parameters(model)
                        result.update({"m__n_params": m__n_params})
                        print(":: m__n_params:   ", m__n_params)   

                        # if provided, load a pretrained model
                        if model_weights_init:
                            model.apply(weights_init)
                            print("Randomly initialized model parameters.")
                        # calculate the balancing loss weights if requested (useful for unbalanced datasets)
                        if balance_loss_weights and task_type=='classif':
                            if hasattr(criterion(), 'weight'):
                                lbl_classes = np.unique(y_data)
                                # weight should be calculated as (n_neg_examples/n_pos_examples)
                                class_weights=compute_class_weight('balanced', classes=lbl_classes, y=y_data)
                                print(f":: balance_loss_weights = {class_weights} calculated for classes {lbl_classes} resp.")
                                # update the weights info in the results
                                result['m__criterion']=result['m__criterion'].replace(
                                    '({',"({'weight':"+str(list(class_weights)),1)
                                criterion_params.update({'weight':
                                    torch.tensor(class_weights, dtype=torch.float).cuda()})
                            else:
                                print(f"[WARN]:: balance_loss_weights not calculated since \
{criterion.__name__} loss function doesn't have any 'weight' argument.")

                        criterion_fn = criterion(**criterion_params).cuda()
                        optimizer_fn = optimizer(model.parameters(), **optimizer_params)
                        scheduler_fn = scheduler(optimizer_fn, verbose=debug, **scheduler_params)
                        
                        main_metric = metrics[0].__name__ if metrics else "binary_balanced_accuracy"
                        
                        # configure callbacks 
                        callbacks = []
                        if earlystop_patience:
                            callbacks.extend([
                                EarlyStopping(earlystop_patience, window=2,
                                              ignore_before=earlystop_patience, 
                                              retain_metric="loss", mode='min')]) # dd: do early stopping on the loss
                        if save_model:
                            store_best=True if save_model=='best' else False 
                            callbacks.extend([
                                ModelCheckpoint(path=output_dir, prepend=run_id,
                                                ignore_before=earlystop_patience,
                                                store_best=store_best,
                                                retain_metric=main_metric)])  

                        # create a mask to distinguish between training & validation samples
                        mask = np.ones(len(y_data), dtype=bool)
                        mask[data_val_idx] = False
                        
                        # prepare training and validation data as Pytorch DataLoader objects
                        other_transforms = [ToTensor()]       
                        if rescale_X: other_transforms.insert(0, IntensityRescale(masked=True))
                        transform = transforms.Compose(augmentations + other_transforms)
                        train_data = arrayDataset(X_data[mask], y_data[mask],
                                                  transform=transform, 
                                                  lbl_type=task_type)
                        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) #don't use num_workers as it doesnt go well with the outer parallel jobs, multiprocessing_context='fork') #https://github.com/pytorch/pytorch/issues/44687
                        
                        # for validation ensure the batch size is not bigger than the data size itself
                        batch_size_val = batch_size if batch_size<=len(data_val_idx) else len(data_val_idx) 
                        transform = transforms.Compose(other_transforms)
                        val_data = arrayDatasetWithSubID(X_data[~mask], y_data[~mask],
                                                         i_data[~mask], transform=transform, 
                                                         lbl_type=task_type)
                        val_loader = DataLoader(val_data, batch_size=batch_size_val, shuffle=False, drop_last=False)
                        
                        # in debug mode show a sample MRI before and after augmentations
                        if debug:
                            # plot a sample MRI before augmentation
                            rand_sub_idx = np.random.randint(0, len(y_data[mask]))
                            X_i, y_i = X_data[mask][rand_sub_idx], y_data[mask][rand_sub_idx]
                            
                            save_fig_path = save_fig_prefix+"sample" if save_figures else ''
                            show_MRI_stats(X_i, save_fig_path=save_fig_path,
                                title=f"Sample i={rand_sub_idx}: X shape={X_i.shape},  y={y_i}")
                            # plot the same sample again after augmentation
                            data_sample = train_data[rand_sub_idx]
                            X_i, y_i = data_sample['image'].numpy()[0], data_sample['label'].numpy()
                            
                            
                            save_fig_path = save_fig_prefix+"sample_aug" if save_figures else ''
                            show_MRI_stats(X_i, save_fig_path=save_fig_path,
                                title=f"Sample i={rand_sub_idx} after augmentation: X shape={X_i.shape},  y={y_i}")
                       
                        trainer = Trainer(model,
                                    criterion_fn, optimizer_fn, scheduler=scheduler_fn,
                                    metrics=metrics, callbacks=callbacks,
                                    device=gpu, task_type=task_type)

                        # train model
                        # show gradient flow plot if requested
                        show_grad_flow = False if not show_grad_flow else save_fig_prefix
                        model, report = trainer.train_model(train_loader, val_loader,
                                                          num_epochs=num_epochs,
                                                          show_grad_flow=show_grad_flow)  
                        
                        
                        
                        trainer.visualize_training(report, metrics,
                                                   # save always irrespective of 'save_figures'
                                                   save_fig_path=save_fig_prefix)
                        
                        # Save the training metrics in the results dict
                        for key in ["train_metrics", "val_metrics"]:
                            for metric, value in report[key].items():
                                result[key.replace('_metrics','_curve_')+metric]=value

                        # VALIDATE AGAIN on the val_set to get the predicted probabilities
                        print("----------------------\nRe-evaluatiing on validation data: \n----------------------")
                        trainer.model = report['best_model']
                        (model_outputs, true_labels, 
                         val_report, data_other) = trainer.evaluate_model(val_loader, 
                                                                                     metrics=metrics,
                                                                                     return_results=True,
                                                                                     write_to_dir="") #f"{output_dir}/{run_id}_"                     

                        true_labels = torch.cat(true_labels).detach().float().cpu().numpy().reshape(-1)
                        val_preds = torch.cat(model_outputs).detach().float().cpu().numpy()
                        val_ids = data_other['i']

                        # save the probabilities and true labels in the results
                        result.update({"val_lbls": true_labels.tolist()})            
                        result.update({"val_preds": val_preds.tolist()})
                        result.update({'val_ids': val_ids})
                        
                        result.update({'val_'+k:v[0] for k,v in val_report.items()})
                        
                        # TEST on an independent holdout_set, if available
                        if data_hold:
                            print("----------------------\nEvaluation on holdout data: \n----------------------")
                            if isinstance(data_hold, str) and os.path.exists(data_hold):
                                data_hold = h5py.File(data_hold, 'r')
                                
                            X_data_hold = data_hold[X_name][:]
                            y_data_hold  = data_hold[y_name][:]
                            i_data_hold  = data_hold['i'][:]
                            try:
                                data_hold.close()
                            except:
                                pass
                            del data_hold
                            
                            transform=transforms.Compose(other_transforms)
                            holdout_data = arrayDatasetWithSubID(X_data_hold, y_data_hold, 
                                                                 i_data_hold, transform=transform,
                                                                 lbl_type=task_type)
                            holdout_loader = DataLoader(holdout_data, batch_size=batch_size_val, 
                                                        shuffle=False, drop_last=False)
                            
                            (model_outputs, true_labels, 
                             hold_report, data_other) = trainer.evaluate_model(holdout_loader, 
                                                                     metrics=metrics, 
                                                                     return_results=True,
                                                                     write_to_dir='')#f"{output_dir}/{run_id}_"
                            true_labels = torch.cat(true_labels).float().cpu().numpy().reshape(-1)
                            hold_preds = torch.cat(model_outputs).float().cpu().numpy()
                            hold_ids = data_other['i']
                            # save the probabilities and true labels in the results
                            result.update({'hold_lbls': true_labels.tolist()})
                            result.update({'hold_preds': hold_preds.tolist()})
                            result.update({'hold_ids': hold_ids})
                            
                            result.update({'hold_'+k:v[0] for k,v in hold_report.items()})
                            del holdout_loader
                            
                        if not return_model: del model
                        del optimizer_fn, criterion_fn, scheduler_fn, trainer, transform
                        del train_loader, val_loader, report
                        torch.cuda.empty_cache()
                        FINISHED=True

                    # if OOM occures, clear the GPU memory, reduce the batch_size by 2 and try again
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            # reduce the batch size by 2
                            new_batch_size = batch_size - 2
                            if new_batch_size>0:
                                print(f"[OOM WARN] {e} \n:: reducing the batch_size from {batch_size} to {new_batch_size}")
                                batch_size=new_batch_size
                                print_debug=True
                            else:
                                FINISHED=True
                                print('[OOM ERROR]', str(e), '\nQuitting this run...')
                                print_debug=True
                        else:
                            FINISHED=True
                            raise e
                    
                    # delete all tensors present on the current GPU and free up space
                    clear_reset_gpu(gpu, print_debug)
                    
                # calculate total elapsed runtime
                runtime = datetime.now() - start_time
                result.update({"runtime":int(runtime.total_seconds())})
                 # save result as a dataframe
                pd.DataFrame([result]).to_csv(join(output_dir, run_id+'.csv'))
                print("---------------- CNNpipeline completed ----------------")
                print("RAN FOR {}s".format(str(runtime).split(".")[0]))
                
                if return_model:  return model
###########################################################################################
