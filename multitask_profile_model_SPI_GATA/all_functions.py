## ALL THE FUNCTIONS USED IN pipeline.py
# mostly stuff from the original train_model_example notebook Alex shared with me

## IMPORTS
import sys
# append paths pointing to Data directory on your machine
sys.path.append('/home/katie/bp_repo/multitask_profile_model_SPI_GATA')

import os
import glob
import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfaidx
import pyBigWig
import tqdm
import h5py

import profile_models
from profile_models import place_tensor
import profile_performance

import plotting_helper
from plotting_helper import *

# modify this for your own directory
#os.chdir('/home/katie/bp_repo/multitask_profile_model_SPI_GATA/') # for SPI/GATA models
os.chdir('/home/katie/bp_repo/research') # for CTCF/FOSL2 models
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## DATA LOADING FUNCTIONS
def dna_to_one_hot(seqs):
    """
    Converts a list of DNA ("ACGT") sequences to one-hot encodings, where the
    position of 1s is ordered alphabetically by "ACGT". `seqs` must be a list
    of N strings, where every string is the same length L. Returns an N x L x 4
    NumPy array of one-hot encodings, in the same order as the input sequences.
    All bases will be converted to upper-case prior to performing the encoding.
    Any bases that are not "ACGT" will be given an encoding of all 0s.
    """
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)
    seq_concat = "".join(seqs).upper() + "ACGT"
    one_hot_map = np.identity(5)[:, :-1]
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85
    _, base_inds = np.unique(base_vals, return_inverse=True)
    return one_hot_map[base_inds[:-4]].reshape((len(seqs), seq_len, 4))

class BPDatasetWithoutControls(torch.utils.data.Dataset):
    def __init__(self, coords, batch_size=128, jitter=128, revcomp=True,**kwargs):
        """
        Creates a data loader. `coords` is an N x 3 object array of summit-
        centered coordinates.
        kwargs includes:
        reference_fasta_path (str): path to fasta containing entire genome
        tasks_path (str): path to directory with tasks as subdirectories
        dna_to_one_hot (function): the dna_to_one_hot function defined earlier in this notebook
        tasks (list): list of tasks (strings)
        """
        self.coords = coords
        self.batch_size = batch_size
        self.jitter = jitter
        self.revcomp = revcomp
        for k,v in kwargs.items():
            setattr(self,k,v)

    def __len__(self):
        return int(np.ceil(len(self.coords) / self.batch_size))

    def _get_input_seqs(self, coords_batch):
        """
        For a B x 3 array of coordinates, returns the one-hot-encoded
        sequences as a B x I x 4 array.
        """
        # Pad everything to the right size
        mid = (coords_batch[:, 1] + coords_batch[:, 2]) // 2
        coords_batch[:, 1] = mid - (input_length // 2)
        coords_batch[:, 2] = coords_batch[:, 1] + input_length
        
        # Fetch sequences as strings
        reader = pyfaidx.Fasta(getattr(self,'reference_fasta_path',reference_fasta_path))
        seqs = []
        for chrom, start, end in coords_batch:
            seqs.append(reader[chrom][start:end].seq)
        
        return getattr(self,'dna_to_one_hot',dna_to_one_hot)(seqs)
        
    def _get_profiles_by_task(self, coords_batch, task):
        """
        Per task (a String):
        For a B x 3 array of coordinates, returns the true profiles as a
        B x O x 2 array, and the control profiles as a B x O x 2 array.
        """
        # Pad everything to the right size
        mid = (coords_batch[:, 1] + coords_batch[:, 2]) // 2
        coords_batch[:, 1] = mid - (profile_length // 2)
        coords_batch[:, 2] = coords_batch[:, 1] + profile_length
        
        true_profs = np.empty((len(coords_batch), profile_length, 2))
        
        tf_neg_reader = pyBigWig.open(glob.glob(tasks_path + task + '/' + task + '*neg.bw')[0], "r")
        tf_pos_reader = pyBigWig.open(glob.glob(tasks_path + task + '/' + task + '*pos.bw')[0], "r")
        
        for i, (chrom, start, end) in enumerate(coords_batch):
            true_profs[i, :, 0] = np.nan_to_num(tf_neg_reader.values(chrom, start, end))
            true_profs[i, :, 1] = np.nan_to_num(tf_pos_reader.values(chrom, start, end))

        tf_neg_reader.close()
        tf_pos_reader.close()
        
        # insert a dimension at axis 1 to allow for concatenation over number of tasks
        return np.expand_dims(true_profs,axis=1)
    
    def _get_profiles(self, coords_batch, tasks):
        """ 
        For a list of tasks (each a String referring to a subdirectory of Data), 
        call _get_profiles_by_task and concatenate all the tasks along axis=1
        """
        true_profs_list = []
        for task in tasks:
            true_profs = self._get_profiles_by_task(coords_batch, task)
            true_profs_list.append(true_profs)
        
        # concatenate over tasks 
        return np.concatenate(true_profs_list,axis=1)
        
    def __getitem__(self, index):
        """
        Returns a batch of data as the input sequences, target profiles,
        and corresponding coordinates for the batch.
        """
        batch_slice = slice(index * self.batch_size, (index + 1) * self.batch_size)
        coords_batch = self.coords[batch_slice]
        
        # Apply jitter
        if self.jitter:
            jitter_amount = np.random.randint(-self.jitter, self.jitter, size=len(coords_batch))
            coords_batch[:, 1] = coords_batch[:, 1] + jitter_amount
            coords_batch[:, 2] = coords_batch[:, 2] + jitter_amount
        
        input_seqs = self._get_input_seqs(coords_batch)
        true_profs = self._get_profiles(coords_batch, getattr(self,'tasks'))
        
        if self.revcomp:
            input_seqs = np.concatenate([input_seqs, np.flip(input_seqs, axis=(1, 2))])
            true_profs = np.concatenate([true_profs, np.flip(true_profs, axis=(2, 3))])  # change axes to 2 and 3
            coords_batch = np.concatenate([coords_batch, coords_batch])
        
        return input_seqs.astype(np.float32), true_profs.astype(np.float32)
    
    def on_epoch_start(self):
        # Shuffle the dataset
        inds = np.random.permutation(len(self.coords))
        self.coords = self.coords[inds]
        
        
class BPDatasetWithControls(BPDatasetWithoutControls):
    def _get_profiles_by_task(self, coords_batch, task):
        """
        Per task (a String):
        For a B x 3 array of coordinates, returns the true profiles as a
        B x O x 2 array, and the control profiles as a B x O x 2 array.
        """
        # Pad everything to the right size
        mid = (coords_batch[:, 1] + coords_batch[:, 2]) // 2
        coords_batch[:, 1] = mid - (profile_length // 2)
        coords_batch[:, 2] = coords_batch[:, 1] + profile_length
        
        true_profs = np.empty((len(coords_batch), profile_length, 2))
        cont_profs = np.empty((len(coords_batch), profile_length, 2))
        
        tf_neg_reader = pyBigWig.open(glob.glob(tasks_path + task + '/' + task + '*neg.bw')[0], "r")
        tf_pos_reader = pyBigWig.open(glob.glob(tasks_path + task + '/' + task + '*pos.bw')[0], "r")
        cont_neg_reader = pyBigWig.open(glob.glob(tasks_path + task + '/control*neg.bw')[0], "r")
        cont_pos_reader = pyBigWig.open(glob.glob(tasks_path + task + '/control*pos.bw')[0], "r")
        
        for i, (chrom, start, end) in enumerate(coords_batch):
            true_profs[i, :, 0] = np.nan_to_num(tf_neg_reader.values(chrom, start, end))
            true_profs[i, :, 1] = np.nan_to_num(tf_pos_reader.values(chrom, start, end))
            cont_profs[i, :, 0] = np.nan_to_num(cont_neg_reader.values(chrom, start, end))
            cont_profs[i, :, 1] = np.nan_to_num(cont_pos_reader.values(chrom, start, end))

        tf_neg_reader.close()
        tf_pos_reader.close()
        cont_neg_reader.close() 
        cont_pos_reader.close()
        
        # insert a dimension at axis 1 to allow for concatenation over number of tasks
        return np.expand_dims(true_profs,axis=1), np.expand_dims(cont_profs,axis=1)
        
    def _get_profiles(self, coords_batch, tasks):
        """ 
        For a list of tasks (each a String referring to a subdirectory of Data), 
        call _get_profiles_by_task and concatenate all the tasks along axis=1
        """
        true_profs_list = []
        cont_profs_list = []
        for task in tasks:
            true_profs, cont_profs = self._get_profiles_by_task(coords_batch, task)
            true_profs_list.append(true_profs)
            cont_profs_list.append(cont_profs)
        
        # concatenate over tasks 
        return np.concatenate(true_profs_list,axis=1), np.concatenate(cont_profs_list,axis=1)
    
    def __getitem__(self, index):
        """
        Returns a batch of data as the input sequences, target profiles,
        control profiles, and corresponding coordinates for the batch.
        """
        batch_slice = slice(index * self.batch_size, (index + 1) * self.batch_size)
        coords_batch = self.coords[batch_slice]
        
        # Apply jitter
        if self.jitter:
            jitter_amount = np.random.randint(-self.jitter, self.jitter, size=len(coords_batch))
            coords_batch[:, 1] = coords_batch[:, 1] + jitter_amount
            coords_batch[:, 2] = coords_batch[:, 2] + jitter_amount
            
        input_seqs = self._get_input_seqs(coords_batch)
        true_profs, cont_profs = self._get_profiles(coords_batch, getattr(self,'tasks'))
        
        if self.revcomp:
            input_seqs = np.concatenate([input_seqs, np.flip(input_seqs, axis=(1, 2))])
            true_profs = np.concatenate([true_profs, np.flip(true_profs, axis=(2, 3))])  # change axes to 2 and 3
            cont_profs = np.concatenate([cont_profs, np.flip(cont_profs, axis=(2, 3))])  # to account for 4D rather than 3D
            coords_batch = np.concatenate([coords_batch, coords_batch])
        
        return input_seqs.astype(np.float32), true_profs.astype(np.float32), cont_profs.astype(np.float32)
    
class BPDatasetWithFakeControls(BPDatasetWithoutControls):
    def _get_profiles_by_task(self, coords_batch, task):
        """
        Per task (a String):
        For a B x 3 array of coordinates, returns the true profiles as a
        B x O x 2 array, and the control profiles as a B x O x 2 array.
        """
        # Pad everything to the right size
        mid = (coords_batch[:, 1] + coords_batch[:, 2]) // 2
        coords_batch[:, 1] = mid - (profile_length // 2)
        coords_batch[:, 2] = coords_batch[:, 1] + profile_length
        
        true_profs = np.empty((len(coords_batch), profile_length, 2))
        cont_profs = np.zeros((len(coords_batch), profile_length, 2))
        
        tf_neg_reader = pyBigWig.open(glob.glob(tasks_path + task + '/' + task + '*neg.bw')[0], "r")
        tf_pos_reader = pyBigWig.open(glob.glob(tasks_path + task + '/' + task + '*pos.bw')[0], "r")
        
        for i, (chrom, start, end) in enumerate(coords_batch):
            true_profs[i, :, 0] = np.nan_to_num(tf_neg_reader.values(chrom, start, end))
            true_profs[i, :, 1] = np.nan_to_num(tf_pos_reader.values(chrom, start, end))

        tf_neg_reader.close()
        tf_pos_reader.close()
        
        # insert a dimension at axis 1 to allow for concatenation over number of tasks
        return np.expand_dims(true_profs,axis=1), np.expand_dims(cont_profs,axis=1)
        
    def _get_profiles(self, coords_batch, tasks):
        """ 
        For a list of tasks (each a String referring to a subdirectory of Data), 
        call _get_profiles_by_task and concatenate all the tasks along axis=1
        """
        true_profs_list = []
        cont_profs_list = []
        for task in tasks:
            true_profs, cont_profs = self._get_profiles_by_task(coords_batch, task)
            true_profs_list.append(true_profs)
            cont_profs_list.append(cont_profs)
        
        # concatenate over tasks 
        return np.concatenate(true_profs_list,axis=1), np.concatenate(cont_profs_list,axis=1)
    
    def __getitem__(self, index):
        """
        Returns a batch of data as the input sequences, target profiles,
        control profiles, and corresponding coordinates for the batch.
        """
        batch_slice = slice(index * self.batch_size, (index + 1) * self.batch_size)
        coords_batch = self.coords[batch_slice]
        
        # Apply jitter
        if self.jitter:
            jitter_amount = np.random.randint(-self.jitter, self.jitter, size=len(coords_batch))
            coords_batch[:, 1] = coords_batch[:, 1] + jitter_amount
            coords_batch[:, 2] = coords_batch[:, 2] + jitter_amount
            
        input_seqs = self._get_input_seqs(coords_batch)
        true_profs, cont_profs = self._get_profiles(coords_batch, getattr(self,'tasks'))
        
        if self.revcomp:
            input_seqs = np.concatenate([input_seqs, np.flip(input_seqs, axis=(1, 2))])
            true_profs = np.concatenate([true_profs, np.flip(true_profs, axis=(2, 3))])  # change axes to 2 and 3
            cont_profs = np.concatenate([cont_profs, np.flip(cont_profs, axis=(2, 3))])  # to account for 4D rather than 3D
            coords_batch = np.concatenate([coords_batch, coords_batch])
        
        return input_seqs.astype(np.float32), true_profs.astype(np.float32), cont_profs.astype(np.float32)
        
def load_task_peak_table(bed_path, task):
    ''' From a bed file path for one task, create a peak table '''
    column_names= [ "chrom", "peak_start", "peak_end", "name", "score",
         "strand", "signal", "pval", "qval", "summit_offset",
                  'i0','i1','i2','i3','i4','i5','i6','i7','i8','i9']  # IDR files have 10 columns of replicate data
    peak_table = pd.read_csv(bed_path,header=None,sep='\t')
    peak_table.columns = column_names[:len(peak_table.columns)]
    peak_table["start"] = peak_table["peak_start"] + peak_table["summit_offset"]
    peak_table["end"] = peak_table["start"] + 1
    peak_table["task"] = task
    peak_table = peak_table.loc[peak_table.chrom != 'chrY']  # get rid of Y and M because they're dumb.
    peak_table = peak_table.loc[peak_table.chrom != 'chrM']
    return peak_table

def load_peak_tables(tasks):
    ''' From a list of tasks (Strings), create a list of separate peak tables and one concatenated peak table '''
    separate_peak_tables = []
    for task in tasks:
        # check if it's a "_copy" task (may be the case in some multi-task control experiments)
        if '_copy' in task:
            bed_path = glob.glob(tasks_path + task[:-5] + '/*bed*',recursive=True)[0]  # remove "_copy" from end of task name
        else:
            bed_path = glob.glob(tasks_path + task + '/*bed*',recursive=True)[0]
        separate_peak_tables.append(load_task_peak_table(bed_path, task))
    return pd.concat(separate_peak_tables)


# Create class for pinning memory when loading batches in parallel
# Use this class for collate_fn argument to torch DataLoader

class SimplePinningBatchWithoutControls:
    def __init__(self, data):
        data_tuple = data[0]
        self.input_seqs = torch.tensor(data_tuple[0]).float()
        self.true_profs = torch.tensor(data_tuple[1]).float()
        
    # custom memory pinning method on custom type
    def pin_memory(self):
        self.input_seqs = self.input_seqs.pin_memory()
        self.true_profs = self.true_profs.pin_memory()
        return self
    
class SimplePinningBatchWithControls(SimplePinningBatchWithoutControls):
    def __init__(self, data):
        super().__init__(data)
        self.cont_profs = torch.tensor(data[0][2]).float()
        
    # custom memory pinning method on custom type
    def pin_memory(self):
        self.input_seqs = self.input_seqs.pin_memory()
        self.true_profs = self.true_profs.pin_memory()
        self.cont_profs = self.cont_profs.pin_memory()
        return self
    

def collate_wrapper_without_controls(batch):
    return SimplePinningBatchWithoutControls(batch)
def collate_wrapper_with_controls(batch):
    return SimplePinningBatchWithControls(batch)

## TRAINING FUNCTIONS - only a few minor changes from Alex's code
def run_epoch(data_loader, model, mode, optimizer=None):
    if mode == "train":
        model.train()
        torch.set_grad_enabled(True)
    
    batch_losses, prof_losses, count_losses = [], [], []
    
    t_iter = tqdm.notebook.tqdm(data_loader, desc="\tLoss: ----")
    for batch in t_iter:
        input_seqs = batch.input_seqs.cuda(device=device, non_blocking=True)
        true_profs = batch.true_profs.cuda(device=device, non_blocking=True)
        cont_profs = batch.cont_profs.cuda(device=device, non_blocking=True) if controls else None
        
        logit_pred_profs, log_pred_counts = model(
            input_seqs,
            cont_profs
        ) if controls else model(input_seqs) 
        
        loss, prof_loss, count_loss = model.correctness_loss(
            true_profs,
            logit_pred_profs, log_pred_counts,
            counts_loss_weight, return_separate_losses=True
        )
          
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()  # Compute gradient
            optimizer.step()  # Update weights through backprop

        batch_losses.append(loss.item())
        prof_losses.append(prof_loss.item())
        count_losses.append(count_loss.item())
        t_iter.set_description(
            "\tLoss: %6.4f" % loss.item()
        )
    torch.cuda.empty_cache()  # EDIT MAR 16
    return batch_losses, prof_losses, count_losses

def train(train_dset, val_dset, train_loader, val_loader, model, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    LOSS_LIST = []  # RETURN THIS
    
    train_losses = {}
    val_losses = {}
    for key in ("batch", "prof", "count"):
        train_losses[key] = []
        val_losses[key] = []

    best_val_loss, best_model_state = float("inf"), model.state_dict()

    for epoch_num in range(num_epochs):
        # permute Datasets inside DataLoaders
        train_dset.on_epoch_start()
        val_dset.on_epoch_start()

        # Train
        batch_losses, prof_losses, count_losses = run_epoch(
            train_loader, model, "train", optimizer
        )

        train_losses["batch"].append(batch_losses)
        train_losses["prof"].append(prof_losses)
        train_losses["count"].append(count_losses)
        #print("Train epoch %d loss: %.6f" % (epoch_num + 1, np.nanmean(batch_losses)))
        LOSS_LIST.append(str(np.nanmean(batch_losses)))

        # Valid
        batch_losses, prof_losses, count_losses = run_epoch(
            val_loader, model, "eval", None
        )

        val_losses["batch"].append(batch_losses)
        val_losses["prof"].append(prof_losses)
        val_losses["count"].append(count_losses)

        val_epoch_loss = np.nanmean(batch_losses)
        #print("Valid epoch %d loss: %.6f" % (epoch_num + 1, val_epoch_loss))
        LOSS_LIST.append(str(np.nanmean(batch_losses)))

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    return LOSS_LIST

## MAIN EVALUATION FUNCTION

# First, define constants and paths
reference_fasta_path = '/home/katie/bp_repo/multitask_profile_model_SPI_GATA/data/genomes/hg38.fasta'

input_length = 2114
profile_length = 1000

train_chroms = ["chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr9", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"]
val_chroms = ["chr8", "chr10"]
test_chroms = ["chr1"]

counts_loss_weight = 20
learning_rate = 0.001

# Actual evaluation function (combines all the functions above)
def evaluate(train_tasks, val_tasks, test_tasks, num_tasks, assay,
             epoch_metrics=False, model_save_path=None):
    ''' 
    Given tasks, return a list with the loss (after each epoch) and performance metrics
    Args:
    train_tasks: list of task names, e.g. ['SPI1', 'GATA2']
    val_tasks: same as train_tasks
    test_tasks: same as train_tasks
    num_tasks: number of tasks (kinda redundant info, I may just remove this later)
    assay: either chip-seq or cutnrun
    
    Optional:
    #controls: (bool) whether or not to use a controls file  # REMOVED MAR 16
    epoch_metrics: (bool) whether or not to save metrics after every epoch
    model_save_path: (str) where to save trained model
    '''
    print(f'num epochs: {num_epochs}')
    print(f'tasks path: {tasks_path}')
    #num_epochs = get_num_epochs()#10 # original 5
    
    #BASE KWARGS
    base_kwargs = {'reference_fasta_path':reference_fasta_path, 
               'tasks_path':tasks_path, 'dna_to_one_hot': dna_to_one_hot}
    
    train_tasks.sort()  # alphabetize everything, to be safe
    val_tasks.sort()
    test_tasks.sort()
    
    train_kwargs = base_kwargs.copy()
    train_kwargs['tasks'] = train_tasks
    
    val_kwargs = base_kwargs.copy()
    val_kwargs['tasks'] = val_tasks
    
    test_kwargs = base_kwargs.copy()
    test_kwargs['tasks'] = test_tasks
    
    train_peak_table = load_peak_tables(train_tasks)
    val_peak_table = load_peak_tables(val_tasks)
    test_peak_table = load_peak_tables(test_tasks)

    # equivalent to all_coords in the original code
    train_tasks_coords = train_peak_table[["chrom", "start", "end"]].values        # don't need task for train or val
    val_tasks_coords = val_peak_table[["chrom", "start", "end"]].values
    test_tasks_coords = test_peak_table[["chrom", "start", "end", "task"]].values  # need task to create statuses

    train_coords = train_tasks_coords [np.isin(train_tasks_coords[:, 0], train_chroms)]
    val_coords = val_tasks_coords [np.isin(val_tasks_coords[:, 0], val_chroms)]   
    test_coords = test_tasks_coords [np.isin(test_tasks_coords[:, 0], test_chroms)]   # keep task column for now

    statuses = []  # only matters for test data

    for i in range(0, len(test_coords), 128):
        batch_statuses = list( test_coords[i: (i + 128) ,3] )
        statuses += batch_statuses * 2  # for reverse-complement augmentation

    if i < len(test_coords):
        statuses += list( test_coords[len(test_coords)%128 + i:,3] ) * 2

    statuses = np.array(statuses)
    model_task_statuses = test_tasks
    test_coords = test_coords[:,:-1]  # get rid of task column now

    # Create Datasets and construct DataLoaders with multiple workers
    num_workers = 15

    BPDataset = BPDatasetWithControls if controls else BPDatasetWithoutControls
    train_dset = BPDataset(train_coords,**train_kwargs)
    val_dset = BPDataset(val_coords,**val_kwargs)
    test_dset = BPDataset(test_coords,**test_kwargs)

    collate_wrapper = collate_wrapper_with_controls if controls else collate_wrapper_without_controls
    train_loader = torch.utils.data.DataLoader(train_dset,num_workers=15,collate_fn=collate_wrapper,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dset,num_workers=15,collate_fn=collate_wrapper,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dset,num_workers=15,collate_fn=collate_wrapper,pin_memory=True)    
    
    ProfilePredictor = profile_models.ProfilePredictorWithControls if controls else profile_models.ProfilePredictorWithoutControls
    model = ProfilePredictor(
        input_length=input_length,
        input_depth=4,
        profile_length=profile_length,
        num_tasks=num_tasks,
        num_strands=2,
        num_dil_conv_layers=9,
        dil_conv_filter_sizes=([21] + ([3] * 8)),
        dil_conv_stride=1,
        dil_conv_dilations=[2 ** i for i in range(9)],
        dil_conv_depth=64,
        prof_conv_kernel_size=75,
        prof_conv_stride=1
    )
        
    model = model.to(device)
    
    combined_loss_and_metrics = [train_tasks, val_tasks, test_tasks]
    
    # train model
    LOSS_LIST = train(train_dset, val_dset, train_loader, val_loader, model, num_epochs)
    
    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
        
    true_profs, log_pred_profs, true_counts, log_pred_counts = get_predictions(test_loader, model)
    
    metrics = profile_performance.compute_performance_metrics(
        true_profs, log_pred_profs, true_counts, log_pred_counts,
        prof_smooth_kernel_sigma=7, prof_smooth_kernel_width=81,
        statuses=statuses, model_task_statuses=model_task_statuses,
        print_updates=False
    )

    FINAL_METRICS = profile_performance.log_performance_metrics(metrics)
    return LOSS_LIST, FINAL_METRICS

global tasks_path
def set_tasks_path(tp):
    global tasks_path
    tasks_path = tp
    
def get_tasks_path():
    return tasks_path

# added March 5, 2022
class DataLoader():
    ''' Load up data! '''
    def __init__(self, tasks, assay, controls, tasks_path, subset, jitter=False, fake_controls=False):
        ''' subset should be a list of some combination of "train", "test", "val", "full" 
            Added March 16: fake_controls is if you need to get predictions using a ChIP-seq model
            trained with controls, but you don't want to use controls (for the March tasks)
            
            IF USING REAL OR FAKE CONTROLS, MUST HAVE CONTROLS=TRUE. Even with fake_controls=True, NEED CONTROLS=TRUE
        ''' 
        set_tasks_path(tasks_path)
        self.tasks = tasks
        self.tasks.sort()    # alphabetize everything, to be safe
        self.assay = assay
        self.controls = controls
        self.tasks_path = tasks_path
        self.subset = subset
        self.jitter = jitter
        self.fake_controls = fake_controls
        set_tasks_path(tasks_path)
        
    def make_loaders(self):
        ''' Create dataloaders ''' 
        kwargs = {'reference_fasta_path':reference_fasta_path, 
               'tasks_path':self.tasks_path, 'dna_to_one_hot': dna_to_one_hot}
        kwargs['tasks'] = self.tasks

        peak_table = load_peak_tables(self.tasks)


        # equivalent to all_coords in the original code
        all_coords = peak_table[["chrom", "start", "end", "task"]].values  # need task to create statuses

        train_coords = all_coords [np.isin(all_coords[:, 0], train_chroms)]
        val_coords = all_coords [np.isin(all_coords[:, 0], val_chroms)]   
        train_coords, val_coords = train_coords[:,:-1], val_coords[:,:-1]   # get rid of task
        test_coords = all_coords [np.isin(all_coords[:, 0], test_chroms)]   # keep task column for now

        statuses = []  # only matters for test data

        for i in range(0, len(test_coords), 128):
            batch_statuses = list( test_coords[i: (i + 128) ,3] )
            statuses += batch_statuses * 2  # for reverse-complement augmentation

        if i < len(test_coords):
            statuses += list( test_coords[len(test_coords)%128 + i:,3] ) * 2

        statuses = np.array(statuses)
        model_task_statuses = self.tasks
        test_coords = test_coords[:,:-1]  # get rid of task column now

        # Create Datasets and construct DataLoaders with multiple workers
        num_workers = 15

        BPDataset = BPDatasetWithControls if self.controls else BPDatasetWithoutControls
        if self.fake_controls:
            BPDataset = BPDatasetWithFakeControls  # added March 16
        train_dset = BPDataset(train_coords,**kwargs)
        val_dset = BPDataset(val_coords,**kwargs)
        test_dset = BPDataset(test_coords,**kwargs)
        full_dset = BPDataset(all_coords[:,:-1],**kwargs)   # need to remove "task" column from all_coords

        collate_wrapper = collate_wrapper_with_controls if self.controls else collate_wrapper_without_controls
        train_loader = torch.utils.data.DataLoader(train_dset,num_workers=15,collate_fn=collate_wrapper,pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dset,num_workers=15,collate_fn=collate_wrapper,pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dset,num_workers=15,collate_fn=collate_wrapper,pin_memory=True)    
        full_loader = torch.utils.data.DataLoader(full_dset,num_workers=15,collate_fn=collate_wrapper,pin_memory=True) 
        
        loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader, 'full': full_loader}
        return dict((i, loaders[i]) for i in self.subset)
    

def save_preds(dataloader, model, outdir, batch_size=128):
    ''' Save predictions in hdf5 file '''
    # Allocate arrays to hold the results
    num_pos = 2 * len(dataloader.dataset.coords) ## for revcomp
    coords_chrom = np.empty(num_pos, dtype=object)
    coords_start = np.empty(num_pos, dtype=int)
    coords_end = np.empty(num_pos, dtype=int)
    
    i = 0
    j = 0

    for batch in dataloader:
        batch_slice_half = slice(i * batch_size, (i + 1) * batch_size)
        batch_slice_full = slice(j * batch_size, (j + 2) * batch_size) 
        # Compute scores
        coords = dataloader.dataset.coords[batch_slice_half]

        # Fill in data - FIXED SEP 25 FOR COORDS_END - COORDS_START to equal 2114, not 1
        coords_chrom[batch_slice_full] = np.concatenate((coords[:, 0], coords[:, 0]))
        
        mid = (coords[:, 1] + coords[:, 2]) // 2   # where 2114 is input length
        coords[:, 1] = mid - (2114 // 2)
        coords[:, 2] = coords[:, 1] + 2114
        
        coords_start[batch_slice_full] = np.concatenate((coords[:, 1], coords[:, 1]))
        coords_end[batch_slice_full] = np.concatenate((coords[:, 2], coords[:, 2]))
        i += 1
        j += 2
    true_profs, log_pred_profs, true_counts, log_pred_counts = get_predictions(dataloader, model)

    # Write to HDF5
    print("Saving result to HDF5...")
    os.makedirs(os.path.dirname(outdir), exist_ok=True)
    with h5py.File(outdir, "w") as f:
        f.create_dataset("coords/coords_chrom", data=coords_chrom.astype("S"))
        f.create_dataset("coords/coords_start", data=coords_start)
        f.create_dataset("coords/coords_end", data=coords_end)
        f.create_dataset("predictions/true_profs", data=true_profs)
        f.create_dataset("predictions/log_pred_profs", data=log_pred_profs)
        f.create_dataset("predictions/true_counts", data=true_counts)
        f.create_dataset("predictions/log_pred_counts", data=log_pred_counts)
