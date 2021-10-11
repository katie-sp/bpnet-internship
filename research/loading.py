import sys
# append paths pointing to data directory on your machine
sys.path.append('/home/katie/bp_repo/research/')
sys.path.append('/home/katie/bp_repo/multitask_profile_model_SPI_GATA/')

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfaidx
import pyBigWig
import tqdm
import glob

import profile_models
from profile_models import place_tensor

def set_tasks_path(tp):
    global tasks_path
    tasks_path = tp
    
def get_tasks_path():
    return tasks_path

class DataLoader():
    ''' Create DataLoaders '''
    def __init__(self, tasks, assay, controls, path, splits, jitter=128):
        '''
        Args:
        tasks (list of str): task names, e.g. ['CTCF_short', 'CTCF_long']
        assay (str): assay name, e.g. 'cutnrun' or 'chip-seq'
        controls (boolean): whether or not controls should be used
        path (str): path to data, e.g. '/home/katie/bp_repo/research/data/cutnrun/'
        splits (list of str): which loaders to create, e.g. ['full', 'train', 'test']
        jitter (int or boolean): jitter amount (default 128) or False if you don't want jitter
        '''
        tasks.sort() # alphabetize, to keep everything ordered the same
        self.tasks = tasks  
        self.assay = 'cutnrun' if 'cut' in assay else 'chip-seq'
        self.controls = controls
        self.path = path
        self.splits = splits
        self.jitter = jitter
        
    def make_loaders(self, return_datasets=False):
        kwargs = base_kwargs.copy()
        kwargs['tasks_path'] = self.path
        kwargs['tasks'] = self.tasks
        
        peak_table = load_peak_tables(self.tasks, self.path)[["chrom", "start", "end"]].values 
        coords = {}
        for split in self.splits:
            coords[split] = peak_table[np.isin(peak_table[:, 0], chroms[split])]
        
        datasets = {}
        BPDataset = BPDatasetWithControls if self.controls else BPDatasetWithoutControls
        for split in self.splits:
            datasets[split] = BPDataset(coords[split], jitter=self.jitter, **kwargs)
        if return_datasets:
            return datasets
        
        loaders = {}
        collate_wrapper = collate_wrapper_with_controls if self.controls else collate_wrapper_without_controls
        for split in self.splits:
            loaders[split] = torch.utils.data.DataLoader(datasets[split],num_workers=15,collate_fn=collate_wrapper,pin_memory=True)
        return loaders
    
    def make_datasets(self):
        return self.make_loaders(return_datasets=True)
    
    def make_statuses(self):
        ''' Returns list of model_task_statuses followed by dictionary of statuses (in appropriate order)''' 
        peak_table = load_peak_tables(self.tasks, self.path)[["chrom", "start", "end", "task"]].values 
        coords = {}
        for split in self.splits:
            coords[split] = peak_table[np.isin(peak_table[:, 0], chroms[split])]
            
        statuses = {}
        for split in self.splits:
            split_statuses = []
            
            for i in range(0, len(coords[split]), 128):
                batch_split_statuses = list( coords[split][i: (i + 128) , 3] )
                split_statuses += batch_split_statuses * 2  # for reverse-complement augmentation

            if i < len(coords[split]):
                split_statuses += list( coords[split][len(coords[split])%128 + i:, 3] ) * 2

            split_statuses = np.array(split_statuses)
            statuses[split] = split_statuses 
        return self.tasks, statuses
    
class ModelLoader():
    ''' Load models '''
    def __init__(self, controls, num_tasks=1, path=None):
        '''
        Args:
        controls (boolean): whether or not controls were used in the model
        num_tasks (int): number of tasks
        path (str): path to model - if None, will generate new model
        '''
        self.path = path
        self.controls = controls
        self.num_tasks = num_tasks
        
    def load_model(self):
        model_type = profile_models.ProfilePredictorWithControls if self.controls else profile_models.ProfilePredictorWithoutControls
        model = model_type(
                                input_length=input_length,
                                input_depth=4,
                                profile_length=profile_length,
                                num_tasks=self.num_tasks,
                                num_strands=2,
                                num_dil_conv_layers=9,
                                dil_conv_filter_sizes=([21] + ([3] * 8)),
                                dil_conv_stride=1,
                                dil_conv_dilations=[2 ** i for i in range(9)],
                                dil_conv_depth=64,
                                prof_conv_kernel_size=75,
                                prof_conv_stride=1
                            ) 
        if self.path is not None:
            model.load_state_dict(torch.load(self.path))
        model.to(device)
        return model
    
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
        
        ** if doing CUT&RUN fragmented tasks, task name MUST be format {tf}_{fl_cutoff}, e.g. CTCF_150
        ** doesn't matter if it's max length 120, min length 150, just make it this format!
        """
        self.coords = coords
        self.batch_size = batch_size
        self.jitter = jitter
        self.revcomp = revcomp
        for k,v in kwargs.items():
            setattr(self,k,v)
            
        # print paths
        for task in self.tasks:
            if '_' in task:
                index = task.index('_')
                fl = task[index + 1:]
                task = task[:index]
            else:
                fl = None
            neg, pos = pathfinder(self.tasks_path, task, 'chip' in self.tasks_path, fl)['tf_neg'], pathfinder(self.tasks_path, task, 'chip' in self.tasks_path, fl)['tf_pos']
            print(f'Negative {task} BigWig: {neg}\nPositive {task} BigWig: {pos}')

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
        B x O x 2 array.
        """
        # Pad everything to the right size
        mid = (coords_batch[:, 1] + coords_batch[:, 2]) // 2
        coords_batch[:, 1] = mid - (profile_length // 2)
        coords_batch[:, 2] = coords_batch[:, 1] + profile_length
        
        true_profs = np.empty((len(coords_batch), profile_length, 2))

        if '_' in task:
            index = task.index('_')
            fl = task[index + 1:]
            task = task[:index]
        else:
            fl = None
        neg, pos = pathfinder(self.tasks_path, task, 'chip' in self.tasks_path, fl)['tf_neg'],pathfinder(self.tasks_path, task, 'chip' in self.tasks_path, fl)['tf_pos']
        tf_neg_reader = pyBigWig.open(neg, "r")
        tf_pos_reader = pyBigWig.open(pos, "r")
        
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
    def __init__(self, coords, batch_size=128, jitter=128, revcomp=True,**kwargs):
        super().__init__(coords, batch_size=128, jitter=128, revcomp=True,**kwargs)
        # check paths
        for task in self.tasks:
            if '_' in task:
                index = task.index('_')
                fl = task[index + 1:]
                task = task[:index]
            else:
                fl = None
            neg, pos= pathfinder(self.tasks_path, task, 'chip' in self.tasks_path, fl)['cont_neg'] ,pathfinder(self.tasks_path, task, 'chip' in self.tasks_path, fl)['cont_pos']
            print(f'Negative {task} CONTROL BigWig: {neg}\nPositive {task} CONTROL BigWig: {pos}')
              
    def _get_profiles_by_task(self, coords_batch, task):
        """
        Per task (a String):
        For a B x 3 array of coordinates, returns the true profiles as a
        B x O x 2 array, and the control profiles as a B x O x 2 array.
        """
        true_profs = super()._get_profiles_by_task(coords_batch, task)
        
        # Pad everything to the right size
        mid = (coords_batch[:, 1] + coords_batch[:, 2]) // 2
        coords_batch[:, 1] = mid - (profile_length // 2)
        coords_batch[:, 2] = coords_batch[:, 1] + profile_length

        cont_profs = np.empty((len(coords_batch), profile_length, 2))
        for task in self.tasks:
            if '_' in task:
                index = task.index('_')
                fl = task[index + 1:]
                task = task[:index]
            else:
                fl = None
            neg, pos = pathfinder(self.tasks_path, task, 'chip' in self.tasks_path, fl)['cont_neg'],pathfinder(self.tasks_path, task, 'chip' in self.tasks_path, fl)['cont_pos']
        cont_neg_reader = pyBigWig.open(neg, "r")
        cont_pos_reader = pyBigWig.open(pos, "r")
        
        for i, (chrom, start, end) in enumerate(coords_batch):
            cont_profs[i, :, 0] = np.nan_to_num(cont_neg_reader.values(chrom, start, end))
            cont_profs[i, :, 1] = np.nan_to_num(cont_pos_reader.values(chrom, start, end))

        cont_neg_reader.close() 
        cont_pos_reader.close()

        # insert a dimension at axis 1 to allow for concatenation over number of tasks
        return true_profs, np.expand_dims(cont_profs,axis=1)
    
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
    
class SimplePinningBatchWithControls:
    def __init__(self, data):
        data_tuple = data[0]
        self.input_seqs = torch.tensor(data_tuple[0]).float()
        self.true_profs = torch.tensor(data_tuple[1]).float()
        self.cont_profs = torch.tensor(data_tuple[2]).float()
        
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

def pathfinder(path, task, controls, fl_cutoff, print_output=False):
    ''' Use glob to find appropriate file paths
        Args:
        path (str): most general path to start searching from, e.g. /data/cutnrun/
        task (str): task name, e.g. CTCF (doesn't matter if there's fragment length info)
        controls (boolean): whether or not to search for and return control file paths
        fl_cutoff (int): what number to search for in bigwigs - specific to short/long CUT&RUN fragment data
        print_output (boolean): whether or not to print paths
        
        Return:
        [bed, neg bw, pos bw, (optionally) neg control bw, pos control bw]
        
        ** requires different fragment length bigwigs to be differentiated by "maxfl120", "minfl150", etc.
    '''    
    returns = {}
    if '_' in task:
        index = task.index('_')
        task = task[:index]
    
    # BED files first
    assert len(glob.glob(path + task + '/*bed*',recursive=True)) == 1, 'Found ' + str(len(glob.glob(path + task + '/*bed*',recursive=True))) +  ' BED files.'
    bed_path = glob.glob(path + task + '/*bed*',recursive=True)[0]
    print(f'{task} BED path: {bed_path}') if print_output else None
    returns['bed'] = bed_path
    
    # TF bigwigs second
    if fl_cutoff is None:
        neg = set(glob.glob(f'{path}{task}/{task}*neg.bw')).symmetric_difference(set(glob.glob(f'{path}{task}/{task}*[0-9]*neg.bw')))
        pos = set(glob.glob(f'{path}{task}/{task}*pos.bw')).symmetric_difference(set(glob.glob(f'{path}{task}/{task}*[0-9]*pos.bw')))
        assert len(neg) == 1, f'Found {len(neg)} neg bigwigs: {neg}';assert len(pos) == 1, f'Found {len(pos)} pos bigwigs: {pos}'
        neg, pos = list(neg)[0], list(pos)[0]
        print(f'Negative {task} BigWig: {neg}\nPositive {task} BigWig: {pos}') if print_output else None
    else:
        # search for the provided fragment length cutoff number in bw file names
        neg, pos = glob.glob(f'{path}{task}/{task}*{fl_cutoff}*neg.bw'), glob.glob(f'{path}{task}/{task}*{fl_cutoff}*pos.bw')
        assert len(neg) == 1, f'Found {len(neg)} neg bigwigs: {neg}';assert len(pos) == 1, f'Found {len(pos)} pos bigwigs: {pos}'
        neg, pos = neg[0], pos[0]
        print(f'Negative {task} BigWig: {neg}\nPositive {task} BigWig: {pos}') if print_output else None
    returns['tf_neg'] = neg;returns['tf_pos'] = pos
        
    # Control bigwigs third (optional) - assumes no issues with fragment length cutoff since CUT&RUN has no controls
    if controls:
        neg, pos = glob.glob(f'{path}{task}/control*neg.bw'), glob.glob(f'{path}{task}/control*pos.bw')
        assert len(neg) == 1,f'Found {len(neg)} neg control bws: {neg}'; assert len(pos) == 1,f'Found {len(pos)} pos control bws: {pos}'
        print(f'Negative Control BigWig ({task}): {neg}\nPositive Control BigWig ({task}): {pos}') if print_output else None
        returns['cont_neg'] = neg[0];returns['cont_pos'] = pos[0]
    
    return returns
    
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
    return peak_table

def load_peak_tables(tasks, tasks_path):
    ''' From a list of tasks (list of str) and the path to the tasks (str), return a list of separate peak tables '''
    separate_peak_tables = []
    for task in tasks:
        bed_path = pathfinder(tasks_path, task, False, None)['bed']
        print(f'{task} BED path: {bed_path}')
        separate_peak_tables.append(load_task_peak_table(bed_path, task))
    return pd.concat(separate_peak_tables)

# modify this for your own directory
os.chdir('/home/katie/bp_repo/research/')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define constants and paths
reference_fasta_path = '/home/katie/bp_repo/multitask_profile_model_SPI_GATA/data/genomes/hg38.fasta'

# Create dictionary with paths to pass as arguments to Dataset constructor
base_kwargs = {'reference_fasta_path':reference_fasta_path}

input_length = 2114
profile_length = 1000

chroms = {}
chroms['full'] = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"]
chroms['train'] = ["chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr9", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"]
chroms['val'] = ["chr8", "chr10"]
chroms['test'] = ["chr1"]

counts_loss_weight = 20
learning_rate = 0.001
num_epochs = 10
