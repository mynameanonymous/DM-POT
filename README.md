# DM-POT: Dynamic Masking with Partial Optimal Transport for Non-stationary Time Series Domain Adaptation

## Requirements
- Python 3
- PyTorch == 1.10
- NumPy == 1.22
- scikit-learn == 1.4.1
- Pandas == 1.3.5
- skorch == 0.15.0
- openpyxl == 3.0.10
- Wandb == 0.16.6

## Datasets

### Available Datasets
We used three public datasets in this study. Preprocessed versions:

- [SSC](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9)
- [UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [MFD](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN)

## Running DM-POT

The experiments are organised hierarchically:
- `--experiment_description` groups several experiments under one directory.
- `--run_description` identifies individual trials within an experiment.

### Training a model

```bash
python trainers/train.py  --experiment_description exp1  \
                --run_description run_1 \
                --da_method DM_POT \
                --dataset HAR \
                --num_runs 3
```

### Launching a sweep

Sweeps are deployed on [Wandb](https://wandb.ai/) for visualization and progress tracking.

```bash
python trainers/train.py  --experiment_description exp1_sweep  \
                --run_description sweep_over_lr \
                --da_method DM_POT \
                --dataset HAR \
                --num_runs 3 \
                --num_sweeps 50 \
                --sweep_project_wandb DM_POT_HAR \
                --is_sweep True
```

Upon the run, you will find the running progress in the specified project page in Wandb.

### Running ablation studies

DM-POT supports ablation experiments via the `--ablations` flag:

```bash
# Run all ablations in sequence
python trainers/train.py  --experiment_description exp1_ablation  \
                --run_description ablation_run \
                --da_method DM_POT \
                --dataset HAR \
                --num_runs 3 \
                --ablations no_masking no_reliability no_pot

# Run a single ablation
python trainers/train.py  --experiment_description exp1_ablation  \
                --run_description ablation_run \
                --da_method DM_POT \
                --dataset HAR \
                --num_runs 3 \
                --ablations no_pot
```

Available ablation options: `none`, `no_masking`, `no_reliability`, `no_pot`.

## Results

- Each run produces cross-domain scenario results in format `src_to_trg_run_x`, where `x` is the run_id (set via `--num_runs`).
- Under each directory, you will find the classification report, a log file, and risk scores. (Uncomment checkpoint saving in `train.py` if needed.)
- After all runs, overall average and std results are saved in the run directory.

## Project Structure

| Directory / File | Description |
|---|---|
| `algorithms/algorithms.py` | DM_POT algorithm class (masking, recovery, POT loss, adaptation) |
| `configs/hparams.py` | Per-dataset hyperparameters for DM_POT |
| `configs/sweep_params.py` | Wandb sweep search spaces per dataset |
| `configs/data_model_configs.py` | Dataset-specific model configurations (channels, classes, scenarios) |
| `models/models.py` | Backbone networks (CNN, VideoMLP), masking functions, signal recovery |
| `models/loss.py` | Loss functions (entropy, POT, contrastive, CORAL, MMD, etc.) |
| `trainers/train.py` | Main training entry point |
| `trainers/abstract_trainer.py` | Base trainer with data loading, metrics, and evaluation |
| `dataloader/dataloader.py` | Data loading and preprocessing |
| `utils.py` | Utilities (logging, metrics, UMAP plotting, weight estimation) |
| `scripts/convert_ta3n_to_pt.py` | Converts TA3N video features to DM-POT `.pt` format |

## Customization

- Hyperparameters: `configs/hparams.py`
- Masking methods: `models/models.py`
- Loss functions: `models/loss.py`
- To add new datasets: add config classes in `configs/data_model_configs.py` and `configs/hparams.py`