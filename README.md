# TRANSFER LEARNING IN 3D CNNs sMRI DATA, A SYSTEMATIC COMPARISON IN THE UKBIOBANK


In this project we will evaluate the following deep learning methods:

ResNet-50:
   1. Pretrained on videos
   2. Pretrained on MRI
   2. Self-supervised with Contrastive Learning
   3. Baseline

And 2 Transfer Learning Approaches: 
    - Fine-tuning
    - Feature Extraction


# Instructions on how to use the project folder

- Folder `dataloaders`: scripts for loading the different datasets from `/ritter/share/data` and creating the `.h5` files. These functions abstract-away the dataset-specifics by creating a dataloader class that handles dataset-specifics within it and creates h5 tables of a standardized format:
- Folder `CNNpipeline`: scripts for running Deep learning CNN model training on the `.h5` files.    
    + To run a training add all models configurations in a .py file in `Config` Folder. Then, run: 
    ` nohup python runCNNpipeline.py 'config.py' &> 'config.py'.out & `
    



