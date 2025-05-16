# Non-stationary Time series domain adaptation

## Requirmenets:
- Python3
- Pytorch==1.10
- Numpy==1.22
- scikit-learn==1.4.1
- Pandas==1.3.5
- skorch==0.15.0 
- openpyxl==3.0.10
- Wandb=0.16.6 

## Datasets

### Available Datasets
We used three public datasets in this study. Whose **preprocessed** versions as follows:

- [SSC](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9)
- [UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [MFD](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN)



## Training procedure

The experiments are organised in a hierarchical way such that:
- Several experiments are collected under one directory assigned by `--experiment_description`.
- Each experiment could have different trials, each is specified by `--run_description`.

### Training a model

To train a model:

```
python trainers/train.py  --experiment_description exp1  \
                --run_description run_1 \
                --da_method TemSR \
                --dataset HAR \
                --num_runs 3 \
```
### Launching a sweep
Sweeps here are deployed on [Wandb](https://wandb.ai/), which makes it easier for visualization, following the training progress, organizing sweeps, and collecting results.

```
python trainers/train.py  --experiment_description exp1_sweep  \
                --run_description sweep_over_lr \
                --da_method TemSR \
                --dataset HAR \
                --num_runs 3\
                --num_sweeps 50 \
                --sweep_project_wandb TemSR_HAR \
                --is_sweep True
```
Upon the run, you will find the running progress in the specified project page in wandb.




## Results

- Each run will have all the cross-domain scenarios results in the format `src_to_trg_run_x`, where `x`
is the run_id (you can have multiple runs by assigning `--num_runs` arg). 
- Under each directory, you will find the classification report, a log file, and the different risks scores. (you can also save the checkpoint by uncomment the code in train.py)
- By the end of the all the runs, you will find the overall average and std results in the run directory.

### Ablation

- For ablation study, you have to change the train.py file with required parameters and loss.
- Further the hyperparameters are in the configs/hparams.py
- different masking methods are in the models/model.py
- all variety of loss are available in the models/loss.py
- If you find any missing loss, or any other method, please feel free to add it and make an issue for it.