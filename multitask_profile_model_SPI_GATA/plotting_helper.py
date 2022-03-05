# recreated march 5, 2022. i am dumb edit nope jk i persevered okkkkk
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
import tqdm
import glob
import pickle

import profile_models
from profile_models import place_tensor
import profile_performance
    
import tqdm
import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt
device = "cuda"

def profile_logits_to_log_probs(logit_pred_profs, axis=2):
    """
    Converts the model's predicted profile logits into normalized log probabilities
    via a softmax on the specified dimension (defaults to axis=2).
    """
    if type(logit_pred_profs) is np.ndarray:
        return logit_pred_profs - \
            scipy.special.logsumexp(logit_pred_profs, axis=axis, keepdims=True)
    else:
        return torch.log_softmax(logit_pred_profs, dim=axis)
    
def get_predictions(data_loader, model):
    all_true_profs, all_log_pred_profs, all_true_counts, all_log_pred_counts = [], [], [], []
    
    for batch in tqdm.notebook.tqdm(data_loader):
        input_seqs = batch.input_seqs.cuda(device=device, non_blocking=True)
        true_profs_np = batch.true_profs.numpy()

        if isinstance(model, profile_models.ProfilePredictorWithControls):
            cont_profs = batch.cont_profs.cuda(device=device, non_blocking=True) 
            logit_pred_profs, log_pred_counts = model(
                input_seqs,
                cont_profs
            )
        
        else: 
            logit_pred_profs, log_pred_counts = model(input_seqs)
        log_pred_profs = profile_logits_to_log_probs(
            logit_pred_profs
        ).detach()
        
        all_true_profs.append(true_profs_np)
        all_log_pred_profs.append(log_pred_profs.cpu().numpy())
        all_true_counts.append(np.sum(true_profs_np, axis=2))
        all_log_pred_counts.append(log_pred_counts.detach().cpu().numpy())

    return np.concatenate(all_true_profs), np.concatenate(all_log_pred_profs), \
        np.concatenate(all_true_counts), np.concatenate(all_log_pred_counts)

class ProfilePlotter():
    def __init__(self, **kwargs):
        ''' 
        KWARGS:
        tasks : (list of str) list of task names
        loaders : (list of DataLoaders) list of TEST loaders in same order as tasks
        models : (list of trained model objects) list of models trained on tasks in same order as tasks
        '''
        for k,v in kwargs.items():
            setattr(self,k,v)
            
        assert len(self.tasks) == len(self.models)
        
        self.info = {}
        for i in range(len(self.tasks)):
            task, loader, model = self.tasks[i], self.loaders[i], self.models[i]
            #true_profs, log_pred_profs, true_counts, log_pred_counts = get_predictions(loader, model)
            self.info[task] = get_predictions(loader, model)
        
    def plot_task(self, task, index, separate_figs=False, titles=True, save_path=None):
        ''' Plot a single task profile - pred vs true 
            Args:
            task : (str) task name
            index : (int) which sample from the peaks to plot
            separate_figs : (bool) whether to separate true vs pred profiles into two plots
            titles : (bool) whether to include a title
            save_path : (str) where to save plots
        '''
        assert task in self.tasks
        true_profs, log_pred_profs, true_counts, log_pred_counts = self.info[task]
        
        pred_profs = np.exp(log_pred_profs[index]) * np.sum(true_profs[index], axis=1, keepdims=True)
        true_profs, true_counts = true_profs[index], true_counts[index]
        
        if self.normalize:
            true_profs = np.divide(true_profs, true_counts + 1)
        
        if not separate_figs:
            fig, ax = plt.subplots(1, figsize=(15, 2))
            ax = [ax]
            if titles:
                plt.title(f'Lighter blue/orange is true profile. Darker blue/orange is predicted profile.\
                Task: {task}, Index: {index}')
            ax[0].plot(true_profs[0,:,0], color="royalblue", alpha=0.5)
            ax[0].plot(-true_profs[0,:,1], color="goldenrod", alpha=0.5)
            ax[0].plot(pred_profs[0,:,0], color="darkslateblue")
            ax[0].plot(-pred_profs[0,:,1], color="darkorange")
        else:
            fig, ax = plt.subplots(2, figsize=(15, 4))
            if titles:
                plt.suptitle(f'True profile on top. Predicted profile on bottom.\
                Task: {task}, Index: {index}')
            ax[0].plot(true_profs[0,:,0], color="royalblue")
            ax[0].plot(-true_profs[0,:,1], color="goldenrod")
            ax[1].plot(pred_profs[0,:,0], color="royalblue")
            ax[1].plot(-pred_profs[0,:,1], color="goldenrod")
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()