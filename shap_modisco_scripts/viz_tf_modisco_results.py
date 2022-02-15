import sys
sys.path.append('/home/katie/bp_repo/shap_modisco_scripts')
import viz_sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np

def viz_importance_scores(shap_scores_path):
    ''' Visualize 5 random importance scores
        Args:
            shap_scores_path : path to hdf5 file containing shap scores
            (can be obtained by running functions in shap_modisco_helper)
    '''
    with h5py.File(shap_scores_path, "r") as f:
        hyp_scores = f["hyp_scores"]
        input_seqs = f["one_hot_seqs"]
        for index in np.random.choice(hyp_scores.shape[0], size=5, replace=False):
            act_scores = hyp_scores[index] * input_seqs[index]
            plt.figure(figsize=(20, 2))
            plt.plot(np.sum(act_scores, axis=1))
            plt.show()
            viz_sequence.plot_weights(act_scores[960 : 1160], subticks_frequency=100)
            
def viz_motif(tfm_results_path): 
    ''' Visualize 5 motifs
        Args:
            tfm_results_path : path to hdf5 file containing TF-modisco results
            (can be obtained by running functions in shap_modisco_helper)
    '''
    with h5py.File(tfm_results_path, "r") as f:
        metaclusters = f["metacluster_idx_to_submetacluster_results"]
        num_metaclusters = len(metaclusters.keys())
        for metacluster_i, metacluster_key in enumerate(metaclusters.keys()):
            metacluster = metaclusters[metacluster_key]
            print("Metacluster: %s (%d/%d)" % (metacluster_key, metacluster_i + 1, num_metaclusters))
            print("==========================================")
            if "patterns" not in metacluster["seqlets_to_patterns_result"].keys():
                continue
            patterns = metacluster["seqlets_to_patterns_result"]["patterns"]
            num_patterns = len(patterns["all_pattern_names"][:])
            for pattern_i, pattern_name in enumerate(patterns["all_pattern_names"][:]):
                pattern_name = pattern_name.decode()
                pattern = patterns[pattern_name]
                seqlets = pattern["seqlets_and_alnmts"]["seqlets"]

                print("Pattern: %s (%d/%d)" % (pattern_name, pattern_i + 1, num_patterns))
                print("--------------------------------------")

                print("%d seqlets" % len(seqlets))
                print("Sequence")
                viz_sequence.plot_weights(pattern["sequence"]["fwd"][:])
                print("Hypothetical contributions")
                viz_sequence.plot_weights(pattern["task0_hypothetical_contribs"]["fwd"][:])
                print("Contribution_scores")
                viz_sequence.plot_weights(pattern["task0_contrib_scores"]["fwd"][:])

                pwm = pattern["sequence"]["fwd"][:]
                act_contribs = pattern["task0_contrib_scores"]["fwd"][:]