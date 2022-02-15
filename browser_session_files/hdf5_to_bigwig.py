import h5py
import pyBigWig
import numpy as np
import tqdm

def import_chrom_sizes(chrom_sizes_tsv):
    """
    From a two-column TSV, imports a list of chromosome names and a parallel
    list of chromosome sizes.
    """
    chroms, sizes = [], []
    with open(chrom_sizes_tsv, "r") as f:
        for line in f:
            chrom, size = line.strip().split("\t")
            chroms.append(chrom)
            sizes.append(int(size))
    return chroms, sizes


def key_str_to_path(key_str):
    """
    Takes a key string as a "/"-delimited string and returns a tuple of keys,
    with "/" removed.
    """
    return tuple(key_str.split("/"))


def import_coords(hdf5_path, chrom_key, start_key, end_key):
    """
    Given the path to an HDF5, and the key paths of where to find the
    chromosome, start, and end coordinates, imports the coordinates as an N x 3
    object array.
    KATIE'S NOTE: both the saved SHAP scores and the saved profiles have the coords saved with them!! just accessed under a certain key!
                  So you don't need any special files or loading, just use the same hdf5 you're using anyway! :]
    """
    with h5py.File(hdf5_path, "r") as f:
        chroms, starts, ends = f, f, f
        for key in chrom_key:
            chroms = chroms[key]
        for key in start_key:
            starts = starts[key]
        for key in end_key:
            ends = ends[key]

        coords = np.empty((len(chroms), 3), dtype=object)
        coords[:, 0] = chroms[:].astype(str)
        coords[:, 1] = starts[:]
        coords[:, 2] = ends[:]
    return coords

def import_imp_scores(hdf5_path, hyp_score_key, one_hot_seq_key):
    """
    Given the path to an HDF5, and the key paths of where to find the
    hypothetical importance scores and one-hot-sequences, imports the actual
    importance scores (collapsed down the base dimension) as an N x L array.
    """
    with h5py.File(hdf5_path, "r") as f:
        hyp_scores, one_hot_seqs = f, f
        for key in hyp_score_key:
            hyp_scores = hyp_scores[key]
        for key in one_hot_seq_key:
            one_hot_seqs = one_hot_seqs[key]

        num_seqs, seq_len = hyp_scores.shape[:2]
        act_scores = np.empty((num_seqs, seq_len))
        
        batch_size = 128
        num_batches = int(np.ceil(num_seqs / batch_size))
        for i in tqdm.trange(num_batches, desc="Importing importance scores"):
            batch = slice(i * batch_size, (i + 1) * batch_size)
            scores = np.sum(hyp_scores[batch] * one_hot_seqs[batch], axis=2)
            act_scores[batch] = scores

    return act_scores


def import_predicted_profiles(hdf5_path, log_pred_prof_key, log_pred_count_key, task_index):
    """
    Given the path to an HDF5, and the key paths of where to find the
    predicted profile log probabilities and the predicted log counts, imports
    the predicted (count-scaled) profiles. Assumes the logarithms are base e.
    Returns an N x L x 2 array.
    task_index should be 0 if single-task 
    """
    with h5py.File(hdf5_path, "r") as f:
        log_pred_profs, log_pred_counts = f, f
        for key in log_pred_prof_key:
            log_pred_profs = log_pred_profs[key]
        for key in log_pred_count_key:
            log_pred_counts = log_pred_counts[key]

        num_seqs, num_tasks, seq_len = log_pred_profs.shape[:3]
        pred_profs = np.empty((num_seqs, seq_len, 2))
        
        batch_size = 128
        num_batches = int(np.ceil(num_seqs / batch_size))
        for i in tqdm.trange(num_batches, desc="Importing predicted profiles"):
            batch = slice(i * batch_size, (i + 1) * batch_size)
            probs = np.exp(log_pred_profs[batch, task_index])
            counts = np.exp(log_pred_counts[batch, task_index]) - 1
            profs = probs * np.expand_dims(counts, axis=1)
            pred_profs[batch] = profs

    return pred_profs


def import_true_profiles(hdf5_path, true_prof_key, task_index):
    """
    Given the path to an HDF5, and the key path of where to find the true
    profiles, imports the true profiles. Returns an N x L x 2 array.
    """
    with h5py.File(hdf5_path, "r") as f:
        true_profs = f
        for key in true_prof_key:
            true_profs = true_profs[key]
        
        num_seqs, num_tasks, seq_len = true_profs.shape[:3]
        profs = np.empty((num_seqs, seq_len, 2))
        
        batch_size = 128
        num_batches = int(np.ceil(num_seqs / batch_size))
        for i in tqdm.trange(num_batches, desc="Importing true profiles"):
            batch = slice(i * batch_size, (i + 1) * batch_size)
            profs[batch] = true_profs[batch, task_index]

    return profs


def write_bigwig(chroms, chrom_sizes, coords, values, bw_path):
    """
    Writes a BigWig at `bw_path`, where `coords` contains the given `values`.
    `chroms` and `chrom_sizes` set the boundaries. Note that `values` should be
    an N x L array, and `coords` should be an N x 3 object array.
    Write each strand separately
    """
    bw = pyBigWig.open(bw_path, "w")

    # Write the header
    header = [
        (chrom, size) for chrom, size in zip(chroms, chrom_sizes)
    ]
    bw.addHeader(header)

    # Re-pad coordinates if needed to be the right size
    midpoints = (coords[:, 1] + coords[:, 2]) // 2
    coords[:, 1] = midpoints - (values.shape[1] // 2)
    coords[:, 2] = coords[:, 1] + values.shape[1]

    # Sort coordinate indices, first by chromosome, then by start coordinate
    chrom_inds = {chroms[i] : i for i in range(len(chroms))}
    sorted_inds = sorted(
        np.arange(len(coords)),
        key=(lambda i: (chrom_inds[coords[i, 0]], coords[i, 1]))
    )

    # Fill in the values
    curr_chrom, last_end = None, 0
    for i in tqdm.tqdm(sorted_inds, desc="Writing BigWig"):
        chrom, start, end = coords[i]
        vals = values[i]

        if chrom != curr_chrom:
            curr_chrom, last_end = chrom, 0

        if last_end >= start + len(vals):
            # We've already written past the entire coordinate; skip this one
            continue
        if last_end > start:
            # We've partially written into this coordinate
            start_range = np.arange(last_end, end)
            vals = vals[last_end - start:]
        else:
            # No overlap, take the whole set of values
            start_range = np.arange(start, end)

        bw.addEntries(
            [chrom] * len(vals), start_range.tolist(), ends=(start_range + 1).tolist(), values=vals.tolist()
        )  # cast arrays as lists because my pyBigWig was compiled without numpy support
        last_end = end

    bw.close()


if __name__ == "__main__":
    import argparse
    chrom_sizes_tsv = "/home/katie/bp_repo/research/data/hg38.chrom.sizes"
    
    parser = argparse.ArgumentParser(description='HDF5 to BigWig')
    parser.add_argument('--hdf5_path', '-p', type=str, help='Full path to hdf5 file (.h5)') 
    parser.add_argument('--bw_path', '-o', type=str, help='Full path to output bw file to SAVE results (.bw) DO NOT INCLUDE THE .bw PLEASEEEEEE')
    parser.add_argument('--value_type', '-v', type=str,
                        help='Type of data to be saved. Options are: imp_scores, predicted_profs, true_profs')
    parser.add_argument('--task_index', '-t', type=int, default=0,
                        help='Task index, only for multi-task model predictions. Default 0')
    
    args = parser.parse_args() 
    hdf5_path = args.hdf5_path
    bw_path = args.bw_path
    value_type = args.value_type
    task_index = args.task_index
    assert value_type in ['imp_scores', 'predicted_profs', 'true_profs']
    assert task_index in [0, 1]
    
    chroms, chrom_sizes = import_chrom_sizes(chrom_sizes_tsv)
    
    chrom_key, start_key, end_key = ['coords_chrom'],['coords_start'],['coords_end']
    if value_type != 'imp_scores':       # need to add on to the path to access the coords in the saved profile hdf5 files
        chrom_key.insert(0, 'coords')
        start_key.insert(0, 'coords')
        end_key.insert(0, 'coords')
    coords = import_coords(
        hdf5_path,
        chrom_key=chrom_key, 
        start_key=start_key,
        end_key=end_key
    )

    assert np.all(coords[:,2] - coords[:,1] == 2114)  # make sure coords are all length 2114
    
    if value_type == 'imp_scores':
        values = import_imp_scores(
            hdf5_path,
            hyp_score_key=['hyp_scores'], 
            one_hot_seq_key=['one_hot_seqs']
        )
    elif value_type == 'predicted_profs':
        values = import_predicted_profiles(hdf5_path, 
                       log_pred_prof_key=['predictions', 'log_pred_profs'], 
                       log_pred_count_key=['predictions', 'log_pred_counts'],
                       task_index=task_index)
    else:
        values = import_true_profiles(
            hdf5_path, 
            true_prof_key=['predictions', 'true_profs'], 
            task_index=task_index)
        
    if value_type != 'imp_scores':  # importance scores not stranded BUT PROFILES STRANDED
        values0 = values[:,:,0]
        values1 = values[:,:,1]
        assert not np.array_equal(values0, values1)
        write_bigwig(chroms, chrom_sizes, coords, values0, bw_path + '_strand0.bw')
        write_bigwig(chroms, chrom_sizes, coords, values1, bw_path + '_strand1.bw')
    else:
        write_bigwig(chroms, chrom_sizes, coords, values, bw_path + '.bw')