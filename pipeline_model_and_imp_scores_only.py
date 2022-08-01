import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import sys
sys.path.append('/home/katie/bp_repo/research/')
sys.path.append('/home/katie/bp_repo/multitask_profile_model_SPI_GATA/')
sys.path.append('/home/katie/bp_repo/')
sys.path.append('/home/katie/bp_repo/shap_modisco_scripts/')
sys.path.append('/home/katie/bp_repo/shap/')

import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfaidx
import pyBigWig
import tqdm
import glob
import pickle

import all_functions 
from all_functions import *

import profile_models
from profile_models import place_tensor, ModelLoader
import profile_performance

import plotting_helper
from plotting_helper import *

import viz_sequence
import compute_shap
from importlib import reload

import modisco
import h5py

import shap_modisco_helper
from shap_modisco_helper import *

import viz_tf_modisco_results
from viz_tf_modisco_results import *

os.chdir('/home/katie/bp_repo/research/')

device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

# FIX TQDM.NOTEBOOK
if not hasattr(tqdm,'notebook'):
    tqdm.notebook = tqdm.std

if __name__ == "__main__":
    # sys.stdout = open('/home/katie/bp_repo/pipeline_outputs/stdout.txt', 'a')
    # sys.stderr = open('/home/katie/bp_repo/pipeline_outputs/stderr.txt', 'a')
    
    import argparse
    # SET VARIABLES
    
    # assay : "chip-seq" or "cutnrun"
    # tasks : list, like ["CTCF"] or ["CTCF", "FOSL2"] 
    # controls: True for ChIP-seq, False for CUT&RUN
    # outdir : where to save all outputs, like '/home/katie/bp_repo/pipeline_outputs/fosl2_chipseq_oct3/'
    # epochs: number of epochs to train model. Default 10
    # metrics: whether or not to save metrics after each epoch of training
    parser = argparse.ArgumentParser(description='Pipeline')
    
    parser.add_argument('--assay', '-a', type=str,
                        help='Assay type (chip-seq or cutnrun)')
    parser.add_argument('--tasks', '-t', nargs='+',
                        help='list, like CTCF or CTCF FOSL2')
    parser.add_argument('--controls', '-c', action='store_true',
                        help='SWITCH! If NOT included, no controls are used. If -c included, controls used.')
    parser.add_argument('--outdir', '-o', type=str,
                        help='Output directory, like /home/katie/bp_repo/pipeline_outputs/fosl2_chipseq_oct3/')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='Number of epochs to train model. Default 10')
    parser.add_argument('--metrics', '-m', action='store_true',
                        help='SWITCH! If NOT included, epoch metrics are NOT saved. If -m included, epoch metrics are saved.')
    parser.add_argument('--learning-rate', '-l', type=int, default=0.001,
                        help='Log learning rate; should be between -5 and -1 (then lr = 10^lr). Default 0.001 (-3)')
    parser.add_argument('--counts-loss-weight', '-w', type=int, default=20,
                        help='Log counts loss weight; should be between 0 and 3 (then clw = 10^clw). Default 20')
    
    args = parser.parse_args()
    assay = args.assay
    tasks = sorted(args.tasks)
    controls = args.controls
    outdir = args.outdir
    num_epochs = args.epochs
    epoch_metrics = args.metrics
    num_tasks = len(tasks)
    learning_rate = args.learning_rate
    counts_loss_weight = args.counts_loss_weight
    
    from datetime import date
    INFO = f'Assay: {assay}\nTasks: {tasks}\nNumber of tasks: {num_tasks}\nControls: {controls}\nOutput directory: {outdir}\nNumber of epochs: {num_epochs}\nEpoch metrics saved: {epoch_metrics}\nLearning rate: {learning_rate}\nCounts loss weight: {counts_loss_weight} or {10**counts_loss_weight}'
    sys.stderr.write(f'\n-------------------------------------\nDate: {date.today().strftime("%B %d, %Y")}\n{INFO}\n')
    sys.stdout.write(f'\n-------------------------------------\nDate: {date.today().strftime("%B %d, %Y")}\n{INFO}\n')
    
    os.makedirs(outdir, exist_ok=True)

    tasks_path = f'/home/katie/bp_repo/research/data/{assay}/'
    all_functions.tasks_path = tasks_path
    all_functions.num_epochs = num_epochs
    all_functions.controls = controls

    # TRAINING
    losses, metrics = evaluate(tasks, tasks, tasks, num_tasks, assay,
                               epoch_metrics, model_save_path=outdir + 'model.state_dict',
                               counts_loss_weight=10**counts_loss_weight, learning_rate=10**learning_rate)
    
    pickle.dump(metrics, open(outdir + 'metrics.pkl', 'wb'))
    pickle.dump(losses, open(outdir + 'losses.pkl', 'wb'))

    # SAVE PREDICTIONS
    model = ModelLoader(outdir + 'model.state_dict', controls, num_tasks).load_model()
    full_dataloader = DataLoader(tasks, assay, controls, tasks_path, ['full'], jitter=False).make_loaders()['full']
    save_preds(full_dataloader, model, outdir + 'preds')

    # DeepSHAP
    control_type = 'matched' if controls else None
    if num_tasks == 1:
        make_shap_scores(outdir + 'model.state_dict',
                        'profile', full_dataloader, 2114, num_tasks, outdir + 'shap_scores',
                        '/home/katie/bp_repo/multitask_profile_model_SPI_GATA/data/genomes/hg38.fasta',
                        '/home/katie/bp_repo/research/data/hg38.chrom.sizes', controls=control_type)
    else:
        make_shap_scores(outdir + 'model.state_dict',
                        'profile', full_dataloader, 2114, num_tasks, outdir + 'shap_scores_' + tasks[0],
                        '/home/katie/bp_repo/multitask_profile_model_SPI_GATA/data/genomes/hg38.fasta',
                        '/home/katie/bp_repo/research/data/hg38.chrom.sizes', 
                         task_index=0, controls=control_type)
        make_shap_scores(outdir + 'model.state_dict',
                        'profile', full_dataloader, 2114, num_tasks, outdir + 'shap_scores_' + tasks[1],
                        '/home/katie/bp_repo/multitask_profile_model_SPI_GATA/data/genomes/hg38.fasta',
                        '/home/katie/bp_repo/research/data/hg38.chrom.sizes', 
                         task_index=1, controls=control_type)
