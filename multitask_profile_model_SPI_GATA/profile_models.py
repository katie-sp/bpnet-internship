# Author: Alex Tseng (amtseng@stanford.edu)

import torch

def place_tensor(tensor):
    """
    Places a tensor on GPU, if PyTorch sees CUDA; otherwise, the returned tensor
    remains on CPU.
    """
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def multinomial_log_probs(category_log_probs, trials, query_counts):
    """
    Defines multinomial distributions and computes the probability of seeing
    the queried counts under these distributions. This defines D different
    distributions (that all have the same number of classes), and returns D
    probabilities corresponding to each distribution.
    Arguments:
        `category_log_probs`: a D x N tensor containing log probabilities (base
            e) of seeing each of the N classes/categories
        `trials`: a D-tensor containing the total number of trials for each
            distribution (can be different numbers)
        `query_counts`: a D x N tensor containing the observed count of each
            category in each distribution; the probability is computed for these
            observations
    Returns a D-tensor containing the log probabilities (base e) of each
    observed query with its corresponding distribution. Note that D can be
    replaced with any shape (i.e. only the last dimension is reduced).
    """
    # Multinomial probability = n! / (x1!...xk!) * p1^x1 * ... pk^xk
    # Log prob = log(n!) - (log(x1!) ... + log(xk!)) + x1log(p1) ... + xklog(pk)
    trials, query_counts = trials.float(), query_counts.float()
    log_n_fact = torch.lgamma(trials + 1)
    log_counts_fact = torch.lgamma(query_counts + 1)
    log_counts_fact_sum = torch.sum(log_counts_fact, dim=-1)
    log_prob_pows = category_log_probs * query_counts  # Elementwise sum
    log_prob_pows_sum = torch.sum(log_prob_pows, dim=-1)

    return log_n_fact - log_counts_fact_sum + log_prob_pows_sum


class ProfilePredictor(torch.nn.Module):
    def __init__(self):
        """
        Base class of profile models. Implements the loss function for the BPNet
        architecture, as described here:
            https://www.nature.com/articles/s41588-021-00782-6
        """
        super().__init__()

	# MSE Loss for counts
        self.mse_loss = torch.nn.MSELoss(reduction="none")
    
    def correctness_loss(
        self, true_profs, logit_pred_profs, log_pred_counts, count_loss_weight,
        return_separate_losses=False
    ):
        """
        Returns the loss of the correctness off the predicted profiles and
        predicted read counts. This prediction correctness loss is split into a
        profile loss and a count loss. The profile loss is the -log probability
        of seeing the true profile read counts, given the multinomial
        distribution defined by the predicted profile count probabilities. The
        count loss is a simple mean squared error on the log counts.
        Arguments:
            `true_profs`: a B x T x O x S tensor containing true UNnormalized
                profile values, where B is the batch size, T is the number of
                tasks, O is the profile length, and S is the number of strands;
                the sum of a profile gives the raw read count for that task
            `logit_pred_profs`: a B x T x O x S tensor containing the predicted
                profile _logits_
            `log_pred_counts`: a B x T x S tensor containing the predicted log
                read counts (base e)
            `count_loss_weight`: amount to weight the portion of the loss for
                the counts
            `return_separate_losses`: if True, also return the profile and
                counts losses (scalar Tensors)
        Returns a scalar loss tensor, or perhaps 3 scalar loss tensors.
        """
        assert true_profs.size() == logit_pred_profs.size()
        batch_size = true_profs.size(0)
        num_tasks = true_profs.size(1)
        num_strands = true_profs.size(3)

        # Add the profiles together to get the raw counts
        true_counts = torch.sum(true_profs, dim=2)  # Shape: B x T x 2

        # Transpose and reshape the profile inputs from B x T x O x S to
        # B x ST x O; all metrics will be computed for each individual profile,
        # then averaged across pooled tasks/strands, then across the batch
        true_profs = true_profs.transpose(2, 3).reshape(
            batch_size, num_tasks * num_strands, -1
        )
        logit_pred_profs = logit_pred_profs.transpose(2, 3).reshape(
            batch_size, num_tasks * num_strands, -1
        )
        # Reshape the counts from B x T x S to B x ST
        true_counts = true_counts.view(batch_size, num_tasks * num_strands)
        log_pred_counts = log_pred_counts.view(
            batch_size, num_tasks * num_strands
        )

        # 1. Profile loss
        # Compute the log probabilities based on multinomial distributions,
        # each one is based on predicted probabilities, one for each track

        # Convert logits to log probabilities (along the O dimension)
        log_pred_profs = torch.log_softmax(logit_pred_profs, dim=2)

        # Compute probability of seeing true profile under distribution of log
        # predicted probs
        neg_log_likelihood = -multinomial_log_probs(
            log_pred_profs, true_counts, true_profs
        )  # Shape: B x 2T
        # Average across tasks/strands, and then across the batch
        batch_prof_loss = torch.mean(neg_log_likelihood, dim=1)
        prof_loss = torch.mean(batch_prof_loss)

        # 2. Counts loss
        # Mean squared error on the log counts (with 1 added for stability)
        log_true_counts = torch.log(true_counts + 1)
        mse = self.mse_loss(log_pred_counts, log_true_counts)

        # Average across tasks/strands, and then across the batch
        batch_count_loss = torch.mean(mse, dim=1)
        count_loss = torch.mean(batch_count_loss)

        final_loss = prof_loss + (count_loss_weight * count_loss)

        if return_separate_losses:
            return final_loss, prof_loss, count_loss
        else:
            return final_loss



class ProfilePredictorWithoutControls(ProfilePredictor):
    def __init__(
        self, input_length, input_depth, profile_length, num_tasks, num_strands,
        num_dil_conv_layers, dil_conv_filter_sizes, dil_conv_stride,
        dil_conv_dilations, dil_conv_depth, prof_conv_kernel_size,
        prof_conv_stride
    ):
        """
        Creates a profile predictor from a DNA sequence, without control
        profiles.
        Arguments:
            `input_length`: length of the input sequences; each input sequence
                would be D x L, where L is the length
            `input_depth`: depth of the input sequences; each input sequence
                would be D x L, where D is the depth
            `profile_length`: length of the predicted profiles; it must be
                consistent with the convolutional layers specified
            `num_tasks`: number of tasks that are to be predicted; there will be
                two profiles and two read counts predicted for each task
            `num_strands`: number of strands for each profile, typically 1 or 2
            `num_dil_conv_layers`: number of dilating convolutional layers
            `dil_conv_filter_sizes`: sizes of the initial dilating convolutional
                filters; must have `num_conv_layers` entries
            `dil_conv_stride`: stride used for each dilating convolution
            `dil_conv_dilations`: dilations used for each layer of the dilating
                convolutional layers
            `dil_conv_depth`: depth of all the dilating convolutional filters
            `prof_conv_kernel_size`: size of the large convolutional filter used
                for profile prediction
            `prof_conv_stride`: stride used for the large profile convolution

        Creates a close variant of the BPNet architecture, as described here:
            https://www.nature.com/articles/s41588-021-00782-6
        """
        super().__init__()
        
        assert len(dil_conv_filter_sizes) == num_dil_conv_layers
        assert len(dil_conv_dilations) == num_dil_conv_layers

        # Save some parameters
        self.input_depth = input_depth
        self.input_length = input_length
        self.profile_length = profile_length
        self.num_tasks = num_tasks
        self.num_strands = num_strands
        self.num_dil_conv_layers = num_dil_conv_layers
        
        # Convolutional layers with dilations
        self.dil_convs = torch.nn.ModuleList()
        last_out_size = input_length
        for i in range(num_dil_conv_layers):
            kernel_size = dil_conv_filter_sizes[i]
            in_channels = input_depth if i == 0 else dil_conv_depth
            out_channels = dil_conv_depth
            dilation = dil_conv_dilations[i]
            padding = int(dilation * (kernel_size - 1) / 2)  # "same" padding,
                                                             # for easy adding
            self.dil_convs.append(
                torch.nn.Conv1d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, dilation=dilation, padding=padding
                )
            )
          
            last_out_size = last_out_size - (dilation * (kernel_size - 1))
        # The size of the final dilated convolution output, if there _weren't_
        # any padding (i.e. "valid" padding)
        self.last_dil_conv_size = last_out_size

        # ReLU activation for the convolutional layers and attribution prior
        self.relu = torch.nn.ReLU()

        # Profile prediction:
        # Convolutional layer with large kernel
        self.prof_large_conv = torch.nn.Conv1d(
            in_channels=dil_conv_depth,
            out_channels=(num_tasks * num_strands),
            kernel_size=prof_conv_kernel_size
        )

        self.prof_pred_size = self.last_dil_conv_size - \
            (prof_conv_kernel_size - 1)

        assert self.prof_pred_size == profile_length, \
            "Prediction length is specified to be %d, but with the given " +\
            "input length of %d and the given convolutions, the computed " +\
            "prediction length is %d" % \
            (profile_length, input_length, self.prof_pred_size)

        # Length-1 convolution over the convolutional output to get the final
        # profile
        self.prof_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * num_strands),
            out_channels=(num_tasks * num_strands),
            kernel_size=1, groups=num_tasks  # One set of filters over each task
        )
        
        # Counts prediction:
        # Global average pooling
        self.count_pool = torch.nn.AvgPool1d(
            kernel_size=self.last_dil_conv_size
        )

        # Dense layer to consolidate pooled result to small number of features
        self.count_dense = torch.nn.Linear(
            in_features=dil_conv_depth,
            out_features=(num_tasks * num_strands)
        )

        # Dense layer over pooling features to get the final counts, implemented
        # as grouped convolution with kernel size 1
        self.count_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * num_strands),
            out_channels=(num_tasks * num_strands),
            kernel_size=1, groups=num_tasks
        )
 
    def forward(self, input_seqs):
        """
        Computes a forward pass on a batch of sequences.
        Arguments:
            `inputs_seqs`: a B x I x D tensor, where B is the batch size, I is
                the input sequence length, and D is the number of input channels
        Returns the predicted profiles (unnormalized logits) for each task and
        each strand (a B x T x O x S tensor), and the predicted log
        counts (base e) for each task and each strand (a B x T x S) tensor.
        """
        batch_size = input_seqs.size(0)
        input_length = input_seqs.size(1)
        assert input_length == self.input_length

        # PyTorch prefers convolutions to be channel first, so transpose the
        # input
        input_seqs = input_seqs.transpose(1, 2)  # Shape: B x D x I

        # 1. Perform dilated convolutions on the input, each layer's input is
        # the sum of all previous layers' outputs
        dil_conv_out_list = None
        dil_conv_sum = 0
        for i, dil_conv in enumerate(self.dil_convs):
            if i == 0:
                dil_conv_out = self.relu(dil_conv(input_seqs))
            else:
                dil_conv_out = self.relu(dil_conv(dil_conv_sum))
            
            dil_conv_sum = dil_conv_out + dil_conv_sum

        # 2. Truncate the final dilated convolutional layer output so that it
        # only has entries that did not see padding; this is equivalent to
        # truncating it to the size it would be if no padding were ever added
        start = int((dil_conv_out.size(2) - self.last_dil_conv_size) / 2)
        end = start + self.last_dil_conv_size
        dil_conv_out_cut = dil_conv_sum[:, :, start : end]

        # Branch A: profile prediction
        # A1. Perform convolution with a large kernel
        prof_large_conv_out = self.prof_large_conv(dil_conv_out_cut)
        # Shape: B x ST x O

        # A2. Perform length-1 convolutions over the profiles; there are T
        # convolutions, each one is done over one set of prof_large_conv_out
        prof_one_conv_out = self.prof_one_conv(prof_large_conv_out)
        # Shape: B x ST x O
        prof_pred = prof_one_conv_out.view(
            batch_size, self.num_tasks, self.num_strands, -1
        )
        # Transpose profile predictions to get B x T x O x S
        prof_pred = prof_pred.transpose(2, 3)
        
        # Branch B: read count prediction
        # B1. Global average pooling across the output of dilated convolutions
        count_pool_out = self.count_pool(dil_conv_out_cut)  # Shape: B x X x 1
        count_pool_out = torch.squeeze(count_pool_out, dim=2)

        # B2. Reduce pooling output to fewer features, a pair for each task
        count_dense_out = self.count_dense(count_pool_out)  # Shape: B x ST
        count_dense_out = count_dense_out.view(
            batch_size, self.num_strands * self.num_tasks, 1
        )

        # B3. Dense layer over the last layer's outputs; each set of counts gets
        # a different dense network (implemented as convolution with kernel size
        # 1)
        count_one_conv_out = self.count_one_conv(count_dense_out)
        # Shape: B x ST x 1
        count_pred = count_one_conv_out.view(
            batch_size, self.num_tasks, self.num_strands, -1
        )
        # Shape: B x T x S x 1
        count_pred = torch.squeeze(count_pred, dim=3)  # Shape: B x T x S

        return prof_pred, count_pred

    
class ProfilePredictorWithControls(ProfilePredictor):
    def __init__(
        self, input_length, input_depth, profile_length, num_tasks, num_strands,
        num_dil_conv_layers, dil_conv_filter_sizes, dil_conv_stride,
        dil_conv_dilations, dil_conv_depth, prof_conv_kernel_size,
        prof_conv_stride
    ):
        """
        Creates a profile predictor from a DNA sequence, using control profiles.
        Arguments:
            `input_length`: length of the input sequences; each input sequence
                would be D x L, where L is the length
            `input_depth`: depth of the input sequences; each input sequence
                would be D x L, where D is the depth
            `profile_length`: length of the predicted profiles; it must be
                consistent with the convolutional layers specified
            `num_tasks`: number of tasks that are to be predicted; there will be
                two profiles and two read counts predicted for each task
            `num_strands`: number of strands for each profile, typically 1 or 2
            `num_dil_conv_layers`: number of dilating convolutional layers
            `dil_conv_filter_sizes`: sizes of the initial dilating convolutional
                filters; must have `num_conv_layers` entries
            `dil_conv_stride`: stride used for each dilating convolution
            `dil_conv_dilations`: dilations used for each layer of the dilating
                convolutional layers
            `dil_conv_depth`: depth of all the dilating convolutional filters
            `prof_conv_kernel_size`: size of the large convolutional filter used
                for profile prediction
            `prof_conv_stride`: stride used for the large profile convolution

        Creates a close variant of the BPNet architecture, as described here:
            https://www.nature.com/articles/s41588-021-00782-6
        """
        super().__init__()
        
        assert len(dil_conv_filter_sizes) == num_dil_conv_layers
        assert len(dil_conv_dilations) == num_dil_conv_layers

        # Save some parameters
        self.input_depth = input_depth
        self.input_length = input_length
        self.profile_length = profile_length
        self.num_tasks = num_tasks
        self.num_strands = num_strands
        self.num_dil_conv_layers = num_dil_conv_layers
        
        # Convolutional layers with dilations
        self.dil_convs = torch.nn.ModuleList()
        last_out_size = input_length
        for i in range(num_dil_conv_layers):
            kernel_size = dil_conv_filter_sizes[i]
            in_channels = input_depth if i == 0 else dil_conv_depth
            out_channels = dil_conv_depth
            dilation = dil_conv_dilations[i]
            padding = int(dilation * (kernel_size - 1) / 2)  # "same" padding,
                                                             # for easy adding
            self.dil_convs.append(
                torch.nn.Conv1d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, dilation=dilation, padding=padding
                )
            )
          
            last_out_size = last_out_size - (dilation * (kernel_size - 1))
        # The size of the final dilated convolution output, if there _weren't_
        # any padding (i.e. "valid" padding)
        self.last_dil_conv_size = last_out_size

        # ReLU activation for the convolutional layers and attribution prior
        self.relu = torch.nn.ReLU()

        # Profile prediction:
        # Convolutional layer with large kernel
        self.prof_large_conv = torch.nn.Conv1d(
            in_channels=dil_conv_depth,
            out_channels=(num_tasks * num_strands),
            kernel_size=prof_conv_kernel_size
        )

        self.prof_pred_size = self.last_dil_conv_size - \
            (prof_conv_kernel_size - 1)

        assert self.prof_pred_size == profile_length, \
            "Prediction length is specified to be %d, but with the given " +\
            "input length of %d and the given convolutions, the computed " +\
            "prediction length is %d" % \
            (profile_length, input_length, self.prof_pred_size)

        # Length-1 convolution over the convolutional output and controls to
        # get the final profile
        self.prof_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * 2 * num_strands),
            out_channels=(num_tasks * num_strands),
            kernel_size=1, groups=num_tasks  # One set of filters over each task
        )
        
        # Counts prediction:
        # Global average pooling
        self.count_pool = torch.nn.AvgPool1d(
            kernel_size=self.last_dil_conv_size
        )

        # Dense layer to consolidate pooled result to small number of features
        self.count_dense = torch.nn.Linear(
            in_features=dil_conv_depth,
            out_features=(num_tasks * num_strands)
        )

        # Dense layer over pooling features and controls to get the final
        # counts, implemented as grouped convolution with kernel size 1
        self.count_one_conv = torch.nn.Conv1d(
            in_channels=(num_tasks * 2 * num_strands),
            out_channels=(num_tasks * num_strands),
            kernel_size=1, groups=num_tasks
        )
        
	# MSE Loss for counts
        self.mse_loss = torch.nn.MSELoss(reduction="none")

    def forward(self, input_seqs, cont_profs):
        """
        Computes a forward pass on a batch of sequences.
        Arguments:
            `inputs_seqs`: a B x I x D tensor, where B is the batch size, I is
                the input sequence length, and D is the number of input channels
            `cont_profs`: a B x T x O x S tensor, where T is the number of
                tasks, O is the output sequence length, and S is the number of
		strands
        Returns the predicted profiles (unnormalized logits) for each task and
        each strand (a B x T x O x S tensor), and the predicted log counts (base
        e) for each task and each strand (a B x T x S) tensor.
        """
        batch_size = input_seqs.size(0)
        input_length = input_seqs.size(1)
        assert input_length == self.input_length
        assert cont_profs.size(1) == self.num_tasks
        profile_length = cont_profs.size(2)
        assert profile_length == self.profile_length
        num_strands = cont_profs.size(3)
        assert num_strands == self.num_strands

        # PyTorch prefers convolutions to be channel first, so transpose the
        # input and control profiles
        input_seqs = input_seqs.transpose(1, 2)  # Shape: B x D x I
        cont_profs = cont_profs.transpose(2, 3)  # Shape: B x T x S x O

        # Prepare the control tracks: profiles and counts
        cont_counts = torch.sum(cont_profs, dim=3)  # Shape: B x T x 2

        # 1. Perform dilated convolutions on the input, each layer's input is
        # the sum of all previous layers' outputs
        dil_conv_out_list = None
        dil_conv_sum = 0
        for i, dil_conv in enumerate(self.dil_convs):
            if i == 0:
                dil_conv_out = self.relu(dil_conv(input_seqs))
            else:
                dil_conv_out = self.relu(dil_conv(dil_conv_sum))

            dil_conv_sum = dil_conv_out + dil_conv_sum

        # 2. Truncate the final dilated convolutional layer output so that it
        # only has entries that did not see padding; this is equivalent to
        # truncating it to the size it would be if no padding were ever added
        start = int((dil_conv_out.size(2) - self.last_dil_conv_size) / 2)
        end = start + self.last_dil_conv_size
        dil_conv_out_cut = dil_conv_sum[:, :, start : end]

        # Branch A: profile prediction
        # A1. Perform convolution with a large kernel
        prof_large_conv_out = self.prof_large_conv(dil_conv_out_cut)
        # Shape: B x ST x O

        # A2. Concatenate with the control profiles
        # Reshaping is necessary to ensure the tasks are paired adjacently
        prof_large_conv_out = prof_large_conv_out.view(
            batch_size, self.num_tasks, num_strands, -1
        )
        prof_with_cont = torch.cat([prof_large_conv_out, cont_profs], dim=2)
        # Shape: B x T x 2S x O
        prof_with_cont = prof_with_cont.view(
            batch_size, self.num_tasks * 2 * num_strands, -1
        )

        # A3. Perform length-1 convolutions over the concatenated profiles with
        # controls; there are T convolutions, each one is done over one pair of
        # prof_first_conv_out, and a pair of controls
        prof_one_conv_out = self.prof_one_conv(prof_with_cont)
        # Shape: B x ST x O
        prof_pred = prof_one_conv_out.view(
            batch_size, self.num_tasks, num_strands, -1
        )
        # Transpose profile predictions to get B x T x O x S
        prof_pred = prof_pred.transpose(2, 3)
        
        # Branch B: read count prediction
        # B1. Global average pooling across the output of dilated convolutions
        count_pool_out = self.count_pool(dil_conv_out_cut)  # Shape: B x X x 1
        count_pool_out = torch.squeeze(count_pool_out, dim=2)

        # B2. Reduce pooling output to fewer features, a pair for each task
        count_dense_out = self.count_dense(count_pool_out)  # Shape: B x ST

        # B3. Concatenate with the control counts
        # Reshaping is necessary to ensure the tasks are paired adjacently
        count_dense_out = count_dense_out.view(
            batch_size, self.num_tasks, num_strands
        )
        count_with_cont = torch.cat([count_dense_out, cont_counts], dim=2)
        # Shape: B x T x 2S
        count_with_cont = count_with_cont.view(
            batch_size, self.num_tasks * 2 * num_strands, -1
        )  # Shape: B x 2ST x 1

        # B4. Dense layer over the concatenation with control counts; each set
        # of counts gets a different dense network (implemented as convolution
        # with kernel size 1)
        count_one_conv_out = self.count_one_conv(count_with_cont)
        # Shape: B x ST x 1
        count_pred = count_one_conv_out.view(
            batch_size, self.num_tasks, num_strands, -1
        )
        # Shape: B x T x S x 1
        count_pred = torch.squeeze(count_pred, dim=3)  # Shape: B x T x S

        return prof_pred, count_pred


# if __name__ == "__main__":
#     import numpy as np
#     # Here is an example of an instantiated model with typical architectural
#     # parameters
#     model = ProfilePredictorWithControls(
#         input_length=2114,
#         input_depth=4,
#         profile_length=1000,
#         num_tasks=1,
#         num_strands=2,
#         num_dil_conv_layers=9,
#         dil_conv_filter_sizes=([21] + ([3] * 8)),
#         dil_conv_stride=1,
#         dil_conv_dilations=[2 ** i for i in range(9)],
#         dil_conv_depth=64,
#         prof_conv_kernel_size=75,
#         prof_conv_stride=1
#     )

#     # Here is how data can be passed to this model
#     device = torch.device("cuda") if torch.cuda.is_available() \
#         else torch.device("cpu")
#     model = model.to(device)

#     batch_size = 64
#     input_seqs_np = np.random.random((batch_size, 2114, 4))
#     cont_profs_np = np.random.random((batch_size, 1, 1000, 2))
#     true_profs_np = np.random.random((batch_size, 1, 1000, 2))

#     logit_pred_profs, log_pred_counts = model(
#         place_tensor(torch.tensor(input_seqs_np).float()),
#         place_tensor(torch.tensor(cont_profs_np).float())
#     )

#     loss, prof_loss, count_loss = model.correctness_loss(
#         place_tensor(torch.tensor(true_profs_np).float()),
#         logit_pred_profs, log_pred_counts,
#         100, return_separate_losses=True
#     )

#     print(loss.item(), prof_loss.item(), count_loss.item())

    
# KATIE'S ADDITION 
class ModelLoader():
    ''' Class that loads in appropriate model type given path to pretrained model, whether or
        not it was trained using controls, and number of tasks '''
    def __init__(self, model_path, controls=False, num_tasks=1, input_length=2114, profile_length=1000):
        self.model_path = model_path
        self.controls = controls
        self.num_tasks = num_tasks
        self.input_length = input_length
        self.profile_length = profile_length
        
    def load_model(self):
        ''' Load the actual model '''
        model = ProfilePredictorWithControls if self.controls else ProfilePredictorWithoutControls
        model = model(
            input_length=self.input_length,
            input_depth=4,
            profile_length=self.profile_length,
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

        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        model.to('cuda')
        return model