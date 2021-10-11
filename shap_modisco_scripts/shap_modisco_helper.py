# from https://github.com/amtseng/fourier_attribution_priors/blob/481d207499f7164b109db481f3ba989b098fc0e9/src/extract/make_shap_scores.py

import sys
sys.path.append('/home/katie/bp_repo/multitask_profile_model_example')
sys.path.append('/home/katie/bp_repo/shap_modisco_scripts/')
import profile_models
import compute_shap
import torch
import numpy as np
import tqdm
import os
import h5py
import modisco

def make_shap_scores(
    model_path, model_type, dataloader, input_length, num_tasks, out_path,
    reference_fasta, chrom_sizes, task_index=None, profile_length=1000,
    controls=None, num_strands=2, batch_size=128
):
    """
    Computes SHAP scores over an entire dataset, and saves them as an HDF5 file.
    The SHAP scores are computed for all positive input sequences (i.e. peaks or
    positive bins).
    Arguments:
        `model_path`: path to saved model
        `model_type`: either "binary" or "profile"
        `dataloader`: DataLoader of data, which SHAP scores will be computer on
        `input_length`: length of input sequences
        `num_tasks`: number of tasks in the model
        `out_path`: path to HDF5 to save SHAP scores and input sequences
        `reference_fasta`: path to reference FASTA
        `chrom_sizes`: path to chromosome sizes TSV
        `task_index`: index of task to explain; if None, explain all tasks in
            aggregate
        `profile_length`: for profile models, the length of output profiles
        `controls`: for profile models, the kind of controls used: "matched",
            "shared", or None; this also determines the class of the model
        `batch_size`: batch size for SHAP score computation
    Creates/saves an HDF5 containing the SHAP scores and the input sequences.
    The HDF5 has the following keys:
        `coords_chrom`: an N-array of the coordinate chromosomes
        `coords_start`: an N-array of the coordinate starts
        `coords_end`: an N-array of the coordinate ends
        `one_hot_seqs`: an N x I x 4 array of one-hot encoded input sequences
        `hyp_scores`: an N x I x 4 array of hypothetical SHAP contribution
            scores
        `model`: path to the model, `model_path`
    """
    assert model_type in ("binary", "profile")
    
    # Determine the model class and import the model
    if controls == "matched":
        model_class = profile_models.ProfilePredictorWithControls
    elif controls == "shared":
        model_class = profile_models.ProfilePredictorWithSharedControls
    elif controls is None:
        model_class = profile_models.ProfilePredictorWithoutControls
    
    torch.set_grad_enabled(True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model_class(
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
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    num_batches = len(dataloader)

    # Allocate arrays to hold the results
    num_pos = 2 * len(dataloader.dataset.coords) ## for revcomp
    coords_chrom = np.empty(num_pos, dtype=object)
    coords_start = np.empty(num_pos, dtype=int)
    coords_end = np.empty(num_pos, dtype=int)
    one_hot_seqs = np.empty((num_pos, input_length, 4))
    hyp_scores = np.empty((num_pos, input_length, 4))
    
    # Create the explainer
    explainer = compute_shap.create_model_explainer(
        model, 'profile', input_length, profile_length, num_tasks, num_strands,
        controls, task_index=task_index
    )
    i = 0
    j = 0
    # Compute the importance scores
    
    for batch in tqdm.notebook.tqdm(dataloader):
        batch_slice_half = slice(i * batch_size, (i + 1) * batch_size)
        batch_slice_full = slice(j * batch_size, (j + 2) * batch_size) 
        # Compute scores
        coords = dataloader.dataset.coords[batch_slice_half]
        input_seqs = batch.input_seqs.cuda(device=device, non_blocking=True)  
        scores = explainer(
            input_seqs, hide_shap_output=True
        )  # Regardless of the type of controls, we can always put this in

        # Fill in data - FIXED SEP 25 FOR COORDS_END - COORDS_START to equal 2114, not 1
        coords_chrom[batch_slice_full] = np.concatenate((coords[:, 0], coords[:, 0]))
        
        mid = (coords[:, 1] + coords[:, 2]) // 2   # where 2114 is input length
        coords[:, 1] = mid - (2114 // 2)
        coords[:, 2] = coords[:, 1] + 2114
        
        coords_start[batch_slice_full] = np.concatenate((coords[:, 1], coords[:, 1]))
        coords_end[batch_slice_full] = np.concatenate((coords[:, 2], coords[:, 2]))
        one_hot_seqs[batch_slice_full] = input_seqs.cpu()
        hyp_scores[batch_slice_full] = scores
        i += 1
        j += 2

    # Write to HDF5
    print("Saving result to HDF5...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("coords_chrom", data=coords_chrom.astype("S"))
        f.create_dataset("coords_start", data=coords_start)
        f.create_dataset("coords_end", data=coords_end)
        f.create_dataset("hyp_scores", data=hyp_scores)
        f.create_dataset("one_hot_seqs", data=one_hot_seqs)
        model = f.create_dataset("model", data=0)
        model.attrs["model"] = model_path
        
from collections import OrderedDict

def import_shap_scores(shap_scores_hdf5, center_cut_size=None):
    """
    Imports the SHAP scores generated/saved by `make_shap_scores`, and
    returns the hypothetical importance scores, actual importance scores, and
    one-hot encoded sequences.
    Arguments:
        `shap_scores_hdf5`: path to HDF5 of SHAP scores generated by
            `make_shap_scores`
        `center_cut_size`: if specified, keeps only scores/sequences of this
            centered length; by default uses the entire length given in the
            SHAP scores
    Returns the hypothetical importance scores, actual importance scores, and
    corresponding one-hot encoded input sequences. Each is an N x L x 4 array,
    where L is the cut size (or default size).
    """
    score_reader = h5py.File(shap_scores_hdf5, "r")

    # For defining shapes
    num_seqs, input_length, _ = score_reader["hyp_scores"].shape
    if not center_cut_size:
        center_cut_size = input_length
    cut_start = (input_length // 2) - (center_cut_size // 2)
    cut_end = cut_start + center_cut_size

    # For batching up data loading
    batch_size = min(1000, num_seqs)
    num_batches = int(np.ceil(num_seqs / batch_size))

    # Read in hypothetical scores and input sequences in batches
    hyp_scores = np.empty((num_seqs, center_cut_size, 4))
    act_scores = np.empty((num_seqs, center_cut_size, 4))
    one_hot_seqs = np.empty((num_seqs, center_cut_size, 4))

    for i in tqdm.trange(num_batches, desc="Importing SHAP scores"):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        hyp_score_batch = score_reader["hyp_scores"][
            batch_slice, cut_start:cut_end
        ]
        one_hot_seq_batch = score_reader["one_hot_seqs"][
            batch_slice, cut_start:cut_end
        ]
        hyp_scores[batch_slice] = hyp_score_batch
        one_hot_seqs[batch_slice] = one_hot_seq_batch
        act_scores[batch_slice] = hyp_score_batch * one_hot_seq_batch

    score_reader.close()
    
    return hyp_scores, act_scores, one_hot_seqs

def run_tf_modisco(shap_scores_hdf5, outfile, seqlet_outfile, center_cut_size):
    """
    Takes the set of importance scores generated by `make_shap_scores` and
    runs TF-MoDISco on them.
    """
    if center_cut_size:
        center_cut_size = int(center_cut_size)

    hyp_scores, act_scores, input_seqs = import_shap_scores(
        shap_scores_hdf5, center_cut_size
    )
    task_to_hyp_scores, task_to_act_scores = OrderedDict(), OrderedDict()
    task_to_hyp_scores["task0"] = hyp_scores
    task_to_act_scores["task0"] = act_scores

    # Construct workflow pipeline
    tfm_workflow = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
    	sliding_window_size=21,
    	flank_size=10,
        target_seqlet_fdr=0.01,
    	seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
    	    trim_to_window_size=30,
    	    initial_flank_to_add=10,
    	    kmer_len=8,
    	    num_gaps=3,
    	    num_mismatches=2,
    	    final_min_cluster_size=60
    	)
    )

    # Move to output directory to do work
    cwd = os.getcwd()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    os.chdir(os.path.dirname(outfile))

    tfm_results = tfm_workflow(
        task_names=list(task_to_act_scores.keys()),
        contrib_scores=task_to_act_scores,
        hypothetical_contribs=task_to_hyp_scores,
        one_hot=input_seqs
    )

    os.chdir(cwd)
    print("Saving results to %s" % outfile)
    with h5py.File(outfile, "w") as f:
        tfm_results.save_hdf5(f)

    if seqlet_outfile:
        print("Saving seqlets to %s" % seqlet_outfile)
        seqlets = \
            tfm_results.metacluster_idx_to_submetacluster_results[0].seqlets
        bases = np.array(["A", "C", "G", "T"])
        with open(seqlet_outfile, "w") as f:
            for seqlet in seqlets:
                sequence = "".join(
                    bases[np.argmax(seqlet["sequence"].fwd, axis=-1)]
                )
                example_index = seqlet.coor.example_idx
                start, end = seqlet.coor.start, seqlet.coor.end
                f.write(">example%d:%d-%d\n" % (example_index, start, end))
                f.write(sequence + "\n")