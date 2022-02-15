from dinuc_shuffle import dinuc_shuffle
import shap
import torch
import numpy as np
import os
import sys

DEVNULL = open(os.devnull, "w")
STDOUT = sys.stdout

def hide_stdout():
    sys.stdout = DEVNULL
def show_stdout():
    sys.stdout = STDOUT


def place_tensor(tensor):
    """
    Places a tensor on GPU, if PyTorch sees CUDA; otherwise, the returned tensor
    remains on CPU.
    """
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def profile_logits_to_log_probs(logit_pred_profs, axis=2):
    """
    Converts the model's predicted profile logits into normalized probabilities
    via a softmax on the specified dimension (defaults to axis=2).
    Arguments:
        `logit_pred_profs`: a tensor/array containing the predicted profile
            logits
    Returns a tensor/array of the same shape, containing the predicted profiles
    as log probabilities by doing a log softmax on the specified dimension. If
    the input is a tensor, the output will be a tensor. If the input is a NumPy
    array, the output will be a NumPy array. Note that the  reason why this
    function returns log probabilities rather than raw probabilities is for
    numerical stability.
    """
    if type(logit_pred_profs) is np.ndarray:
        return logit_pred_profs - \
            scipy.special.logsumexp(logit_pred_profs, axis=axis, keepdims=True)
    else:
        return torch.log_softmax(logit_pred_profs, dim=axis)


def create_input_seq_background(
    input_seq, input_length, bg_size=10, seed=20200219
):
    """
    From the input sequence to a model, generates a set of background
    sequences to perform interpretation against.
    Arguments:
        `input_seq`: I x 4 tensor of one-hot encoded input sequence, or None
        `input_length`: length of input, I
        `bg_size`: the number of background examples to generate, G
    Returns a G x I x 4 tensor containing randomly dinucleotide-shuffles of the
    original input sequence. If `input_seq` is None, then a G x I x 4 tensor of
    all 0s is returned.
    """
    if input_seq is None:
        input_seq_bg_shape = (bg_size, input_length, 4)
        return place_tensor(torch.zeros(input_seq_bg_shape)).float()

    # Do dinucleotide shuffles
    input_seq_np = input_seq.cpu().numpy()
    rng = np.random.RandomState(seed)
    input_seq_bg_np = dinuc_shuffle(input_seq_np, bg_size, rng=rng)
    return place_tensor(torch.tensor(input_seq_bg_np)).float()


def combine_input_seq_mult_and_diffref(mult, orig_inp, bg_data):
    """
    Computes the hypothetical contribution of any base along the input sequence
    to the final output, given the multipliers for the input sequence
    background. This will simulate all possible base identities as compute a
    "difference-from-reference" for each possible base, averaging the product
    of the multipliers with the differences, over the base identities.
    Arguments:
        `mult`: a G x I x 4 array containing multipliers for the background
            input sequences
        `orig_inp`: the target input sequence to compute contributions for, an
            I x 4 array
        `bg_data`: a G x I x 4 array containing the actual background sequences
    Returns the hypothetical importance scores in an I x 4 array.
    This function is necessary for this specific implementation of DeepSHAP. In
    the original DeepSHAP, the final step is to take the difference of the input
    sequence to each background sequence, and weight this difference by the
    contribution multipliers for the background sequence. However, all
    differences to the background would be only for the given input sequence
    (i.e. the actual importance scores). To get the hypothetical importance
    scores efficiently, we try every possible base for the input sequence, and
    for each one, compute the difference-from-reference and weight by the
    multipliers separately. This allows us to compute the hypothetical scores
    in just one pass, instead of running DeepSHAP many times. To get the actual
    scores for the original input, simply extract the entries for the bases in
    the real input sequence.
    """
    # Reassign arguments to better names; this specific implementation of
    # DeepSHAP requires the arguments to have the above names
    bg_mults, input_seq, bg_seqs = mult, orig_inp, bg_data

    # Allocate array to store hypothetical scores, one set for each background
    # reference (i.e. each difference-from-reference)
    input_seq_hyp_scores_eachdiff = np.empty_like(bg_seqs)
    
    # Loop over the 4 input bases
    for i in range(input_seq.shape[-1]):
        # Create hypothetical input of all one type of base
        hyp_input_seq = np.zeros_like(input_seq)
        hyp_input_seq[:, i] = 1

        # Compute difference from reference for each reference
        diff_from_ref = np.expand_dims(hyp_input_seq, axis=0) - bg_seqs
        # Shape: G x I x 4

        # Weight difference-from-reference by multipliers
        contrib = diff_from_ref * bg_mults

        # Sum across bases axis; this computes the actual importance score AS IF
        # the target sequence were all that base
        input_seq_hyp_scores_eachdiff[:, :, i] = np.sum(contrib, axis=-1)

    # Average hypothetical scores across background
    # references/diff-from-references
    return np.mean(input_seq_hyp_scores_eachdiff, axis=0)


class WrapperProfileModel(torch.nn.Module):
    def __init__(
        self, inner_model, output_head="profile", task_index=None,
        extra_input_shapes=None
    ):
        """
        Takes a profile model and constructs wrapper model around it. This model
        takes in only the input sequence (assumed to be the first argument of
        the inner model) (i.e. input tensor of shape B x I x 4). The model will
        return an output of B x 1, which is either the profile logits (mean-
        normalized and weighted) or the log counts, aggregated to a scalar for
        each input.
        Arguments:
            `inner_model`: a trained `ProfilePredictorWithMatchedControls`,
                `ProfilePredictorWithSharedControls`, or
                `ProfilePredictorWithoutControls`
            `output_head`: either "profile" or "count", which determines which
                output head to explain; defaults to "profile"
            `task_index`: a specific task index (0-indexed) to perform
                explanations from (i.e. explanations will only be from the
                specified outputs); by default explains all tasks in aggregate
            `extra_input_shapes`: if provided, a list of shapes that correspond
                to additional inputs to the inner model; zeros will be filled
                in for these extra inputs; the shapes should not include the
                batch dimension
        """
        assert output_head in ("profile", "count")
        super().__init__()
        self.inner_model = inner_model
        self.task_index = task_index
        self.output_head = output_head
        assert extra_input_shapes is None or type(extra_input_shapes) is list
        self.extra_input_shapes = extra_input_shapes
        
    def forward(self, input_seqs):
        # Run through inner model
        if self.extra_input_shapes:
            extra_inputs = [
                place_tensor(
                    torch.zeros((input_seqs.shape[0],) + input_shape
                )).float() for input_shape in self.extra_input_shapes
            ]
        else:
            extra_inputs = []
        logit_pred_profs, log_pred_counts = self.inner_model(
            *([input_seqs] + extra_inputs)
        )

        if self.output_head == "profile":
            # Explain the mean-normalized logits, weighted by the final
            # probabilities after passing through the softmax; this
            # exponentially increases the weight for high-probability positions,
            # and exponentially reduces the weight for low-probability
            # positions, resulting in a cleaner signal

            # Subtract mean along output profile dimension; this wouldn't change
            # softmax probabilities, but normalizes the magnitude of the logits
            norm_logit_pred_profs = logit_pred_profs - \
                torch.mean(logit_pred_profs, dim=2, keepdim=True)

            # Weight by post-softmax probabilities, but detach it from the graph
            # to avoid explaining those
            pred_prof_log_probs = profile_logits_to_log_probs(
                logit_pred_profs
            ).detach()
            pred_prof_probs = torch.exp(pred_prof_log_probs)
            weighted_norm_logits = norm_logit_pred_profs * pred_prof_probs

            if self.task_index is not None:
                # Subset to specific task
                weighted_norm_logits = \
                    weighted_norm_logits[:, self.task_index : (self.task_index + 1)]
            prof_sum = torch.sum(weighted_norm_logits, dim=(1, 2, 3))

            # DeepSHAP requires the shape to be B x 1
            return torch.unsqueeze(prof_sum, dim=1)
        else:
            if self.task_index is not None:
                log_pred_counts = \
                    log_pred_counts[:, self.task_index : (self.task_index + 1)]
            count_sum = torch.sum(log_pred_counts, dim=(1, 2))
            
            # DeepSHAP requires the shape to be B x 1
            return torch.unsqueeze(count_sum, dim=1)


def create_model_explainer(
    model, model_type, input_length, profile_length=None, num_tasks=None,
    num_strands=None, controls=None, output_head="profile", task_index=None,
    bg_size=10, seed=20200219
):
    """
    Given a trained profile model, creates a Shap DeepExplainer that
    returns hypothetical scores for given input sequences.
    Arguments:
        `model`: a trained `ProfilePredictorWithMatchedControls`,
            `ProfilePredictorWithSharedControls`,
            `ProfilePredictorWithoutControls`
        `model_type`: "profile" 
        `input_length`: length of input sequence, I
        `profile_length`: length of output profiles, O; needed only for profile
            models
        `num_tasks`: number of tasks in model, T; needed only for profile models
        `num_strands`: number of strands in model, T; needed only for profile
            models
        `controls`: the kind of controls used: "matched", "shared", or None;
            if "matched", the control profiles taken in and returned are
            T x O x S; if "shared", the profiles are 1 x O x S; if None, no
            controls need to be provided; needed only for profile models
        `output_head`: either "profile" or "count", which determines which
            output head to explain; defaults to "profile"; needed only for
            profile models
        `task_index`: a specific task index (0-indexed) to perform explanations
            from (i.e. explanations will only be from the specified outputs); by
            default explains all tasks
        `bg_size`: the number of background examples to generate
    Returns a function that takes in input sequences (B x I x 4 array) and
    outputs hypothetical scores for the input sequences (B x I x 4 array).
    """
    assert model_type == "profile"

    assert controls in ("matched", "shared", None)
    if controls == "matched":
        extra_input_shapes = [(num_tasks, profile_length, num_strands)]
    elif controls == "shared":
        extra_input_shapes = [(1, profile_length, num_strands)]
    else:
        extra_input_shapes = None

    assert output_head in ("profile", "count")
    wrapper_model = WrapperProfileModel(
        model, output_head=output_head, task_index=task_index,
        extra_input_shapes=extra_input_shapes
    )

    # DeepSHAP plugs in inputs as lists (even singleton tensors), and expects
    # outputs a lists, so do that here
    def combiner(mult, orig_inp, bg_data):
        # Need to have the named arguments
        return [
            combine_input_seq_mult_and_diffref(
                mult[0], orig_inp[0], bg_data[0]
            )
        ]
    explainer = shap.DeepExplainer(
        model=wrapper_model,
        data=(lambda x: [
            create_input_seq_background(
                x if x is None else x[0], input_length
            )
        ]),
        combine_mult_and_diffref=combiner
    )

    def explain_fn(input_seqs, hide_shap_output=False):
        """
        Given input sequences, returns hypothetical scores for the input
        sequences.
        Arguments:
            `input_seqs`: a B x I x 4 array
            `hide_shap_output`: if True, do not show any warnings from DeepSHAP
        Returns a B x I x 4 array containing hypothetical importance scores for
        each of the B input sequences.
        """
        input_seqs_t = place_tensor(torch.tensor(input_seqs)).float()
        try:
            if hide_shap_output:
                hide_stdout()
            return explainer.shap_values(
                [input_seqs_t], progress_message=None
            )[0]
        except Exception as e:
            raise e
        finally:
            show_stdout()

    return explain_fn


def create_profile_model_explainer(
    model, input_length, profile_length, num_tasks, num_strands, controls,
    output_head="profile", task_index=None, bg_size=10, seed=20200219
):
    """
    Wrapper for `create_model_explainer`.
    """
    return create_model_explainer(
        model, "profile", input_length, profile_length=profile_length,
        num_tasks=num_tasks, num_strands=num_strands, controls=controls,
        output_head=output_head, task_index=task_index, bg_size=bg_size,
        seed=seed
    )

