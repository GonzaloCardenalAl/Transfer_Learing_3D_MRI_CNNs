from os.path import join, dirname
import sys
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# load functions from nitorch
sys.path.append(join(dirname(__file__), "../../../nitorch/"))
import nitorch
from nitorch.transforms import *

# load functions from CNNpipeline
sys.path.append(join(dirname(__file__), "../"))
from models import *
from sklearn.metrics import *

# (0) general local variables created for ease-of-use in the Config class
# the model settings that will be common across all label analysis. 
# This is exists to avoid the need to repeat these settings.
common_model_settings = {
    "model": ResNet50,         
    "batch_size": 20, "num_epochs": 120, "earlystop_patience": 15,
    "optimizer": optim.Adam, "optimizer_params": {"lr": 5e-4, "weight_decay": 1e-4},
    "augmentations": [SagittalFlip(prob=0.5), SagittalTranslate(dist=(-4,4))],
    "scheduler": optim.lr_scheduler.StepLR , "scheduler_params" : {"step_size": 20, "gamma": 0.1},
    "rescale_X": True, 
    "show_grad_flow" : True
    }
    
class Config:
    # (1) DATASET:  Provide the  h5 file containing the data with 'X' and labels
    # Can also provide several h5files with different subsets or different train-holdout splits 
    # but the columns of the h5file (called datasets) must be the same in all of them.
    H5_FILES = [
        "/ritter/share/projects/gonzalo/h5files/h5files5tasks250_moodbalanced.h5", 
        "/ritter/share/projects/gonzalo/h5files/h5files5tasks2000_moodbalanced.h5", 
        "/ritter/share/projects/gonzalo/h5files/h5files5tasks8000_moodbalanced.h5", 
        "/ritter/share/projects/gonzalo/h5files/h5files5tasks35k.h5", 
         "/ritter/share/projects/gonzalo/h5files/h5files5tasks35k_holdout.h5",     
    ]
    

    
    ANALYSIS = {
    # (2) LABELS: Which cols in the h5 files should be used as 'y' / labels in the analysis 
################# Fine-tuning with the parameters loaded from videos ('pretrain' : True, 'feature_extraction' : False (default))
        'sex': 
        # (3) Training settings specific to this label
            dict(
                TASK_TYPE='classif_binary',
                METRICS=[balanced_accuracy_score, accuracy_score],
            # (4) Model configuration to use in this setting
                MODEL_SETTINGS =[
                    {**common_model_settings,
                    "model_name": "ResNet50_videos_finetuning", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r3d50_K_200ep.pth'}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_videos_feature_extraction", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r3d50_K_200ep.pth'}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_MRI_finetuning", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r1d50_AD.pth'}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False},
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_MRI_feature_extraction", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r1d50_AD.pth'}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_baseline", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': False,}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": True},
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_Self_supervised_feature_extraction", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/projects/andreea/ML_for_alcohol_misuse/resnet_cl_25ep_10000s.pth'}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False},
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_Self_supervised_finetuning", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/projects/andreea/ML_for_alcohol_misuse/resnet_cl_25ep_10000s.pth'}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}]       
            ), 
        
        'mood_disorder': 
            dict(
                TASK_TYPE='classif_binary',
                METRICS=[balanced_accuracy_score, accuracy_score],
            # (4) Model configuration to use in this setting
                 MODEL_SETTINGS =[
                    {**common_model_settings,
                    "model_name": "ResNet50_videos_finetuning", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r3d50_K_200ep.pth'}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_videos_feature_extraction", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r3d50_K_200ep.pth'}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                     
                    {**common_model_settings,
                    "model_name": "ResNet50_MRI_finetuning", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r1d50_AD.pth'}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False},
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_MRI_feature_extraction", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r1d50_AD.pth'}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_baseline", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': False,}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": True},
                     
                    {**common_model_settings,
                    "model_name": "ResNet50_Self_supervised_feature_extraction", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/projects/andreea/ML_for_alcohol_misuse/resnet_cl_25ep_10000s.pth'}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False},
                 
                    {**common_model_settings,
                    "model_name": "ResNet50_Self_supervised_finetuning", # unique name to identify this model
                    "model_params":{'out_classes':1, 'task_type':'classif_binary', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/projects/andreea/ML_for_alcohol_misuse/resnet_cl_25ep_10000s.pth'}, 
                    "criterion": nn.BCEWithLogitsLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}]       
            ), 
        
        'alc_int_freq': 
            dict(
                TASK_TYPE='classif',
                METRICS=[accuracy_score],
                MODEL_SETTINGS =[
                    {**common_model_settings,
                    "model_name": "ResNet50_videos_finetuning", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r3d50_K_200ep.pth'}, 
                    "criterion": nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_videos_feature_extraction", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r3d50_K_200ep.pth'}, 
                    "criterion": nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_MRI_finetuning", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r1d50_AD.pth'}, 
                    "criterion": nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False},
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_MRI_feature_extraction", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r1d50_AD.pth'}, 
                    "criterion": nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_baseline", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': False,}, 
                    "criterion": nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": True},
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_Self_supervised_feature_extraction", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/projects/andreea/ML_for_alcohol_misuse/resnet_cl_25ep_10000s.pth'}, 
                    "criterion":nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False},
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_Self_supervised_finetuning", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/projects/andreea/ML_for_alcohol_misuse/resnet_cl_25ep_10000s.pth'}, 
                    "criterion":nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}]   
                ), 
        
        'srt_right_ear_classification': 
            dict(
                TASK_TYPE='classif',
                METRICS=[accuracy_score],
                MODEL_SETTINGS =[
                    {**common_model_settings,
                    "model_name": "ResNet50_videos_finetuning", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r3d50_K_200ep.pth'}, 
                    "criterion": nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_videos_feature_extraction", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r3d50_K_200ep.pth'}, 
                    "criterion": nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_MRI_finetuning", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r1d50_AD.pth'}, 
                    "criterion": nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False},
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_MRI_feature_extraction", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r1d50_AD.pth'}, 
                    "criterion": nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_baseline", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': False,}, 
                    "criterion": nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": True},
                    
                     {**common_model_settings,
                    "model_name": "ResNet50_Self_supervised_feature_extraction", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/projects/andreea/ML_for_alcohol_misuse/resnet_cl_25ep_10000s.pth'}, 
                    "criterion":nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False},
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_Self_supervised_finetuning", # unique name to identify this model
                    "model_params":{'out_classes':3, 'task_type':'classif', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/projects/andreea/ML_for_alcohol_misuse/resnet_cl_25ep_10000s.pth'}, 
                    "criterion":nn.CrossEntropyLoss,
                    "balance_loss_weights":True,
                    "model_weights_init": False}]       
                ), 
        
         'mean_fa_fornix': 
            dict(
                TASK_TYPE='regression',
                METRICS=[explained_variance_score], 
                MODEL_SETTINGS =[
                    {**common_model_settings,
                    "model_name": "ResNet50_videos_finetuning", # unique name to identify this model
                    "rescale_Y" : True,
                    "model_params":{'task_type':'regression', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r3d50_K_200ep.pth'}, 
                    "criterion":nn.MSELoss,"criterion_params":{},
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_videos_feature_extraction", # unique name to identify this model
                    "rescale_Y" : True,
                    "model_params":{'task_type':'regression', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r3d50_K_200ep.pth'}, 
                    "criterion": nn.MSELoss, "criterion_params":{} ,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_MRI_finetuning", # unique name to identify this model
                    "rescale_Y" : True,
                    "model_params":{'out_classes':1, 'task_type':'regression', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r1d50_AD.pth'}, 
                    "criterion": nn.MSELoss, "criterion_params":{} ,
                    "balance_loss_weights":True,
                    "model_weights_init": False},
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_MRI_feature_extraction", # unique name to identify this model
                    "rescale_Y" : True,
                    "model_params":{'out_classes':1, 'task_type':'regression', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/data/UKBB_2020/trained_model/r1d50_AD.pth'}, 
                    "criterion": nn.MSELoss, "criterion_params":{} ,
                    "balance_loss_weights":True,
                    "model_weights_init": False}, 
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_baseline", # unique name to identify this model
                    "rescale_Y" : True,
                    "model_params":{'task_type':'regression', 'freeze_feature_extractor': False,}, 
                    "criterion": nn.MSELoss,"criterion_params":{},
                    "balance_loss_weights":True,
                    "model_weights_init": True},
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_Self_supervised_feature_extraction", # unique name to identify this model
                    "rescale_Y" : True,
                    "model_params":{'task_type':'regression', 'freeze_feature_extractor': True,
                                    'pretrained_model':'/ritter/share/projects/andreea/ML_for_alcohol_misuse/resnet_cl_25ep_10000s.pth'}, 
                    "criterion":nn.MSELoss,"criterion_params":{},
                    "balance_loss_weights":True,
                    "model_weights_init": False},
                    
                    {**common_model_settings,
                    "model_name": "ResNet50_Self_supervised_finetuning", # unique name to identify this model
                    "rescale_Y" : True,
                    "model_params":{'task_type':'regression', 'freeze_feature_extractor': False,
                                    'pretrained_model':'/ritter/share/projects/andreea/ML_for_alcohol_misuse/resnet_cl_25ep_10000s.pth'}, 
                    "criterion":nn.MSELoss,"criterion_params":{},
                    "balance_loss_weights":True,
                    "model_weights_init": False}]
            ),
    }
  ################ Now only trainining the linear layer ('pretrain' : True, 'feature_extraction' : True)
    

   ######### Now the baseline with random weight initialisation
  
    # General run SETTING
    N_CV_TRIALS= 3
    GPUS = [3] 
    RAND_STATE = None
    OUT_FOLDER_SUFFIX = '' # unique name suffix to use on the output dir from this run
