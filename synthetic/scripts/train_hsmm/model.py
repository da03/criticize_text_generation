import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_struct import SemiMarkov
from torch_struct.semirings import MaxSemiring

NEG_INF = -1e6

class HSMMModel(nn.Module):

    def __init__(self, Z, subseq_vocab_sizes, initial_state=0, subseq_pad_id=0, subseq_unk_id=1):
        super().__init__()
        self.Z = Z
        self.initial_state = initial_state
        # Find min_len and max_len
        subseq_min_len = min(subseq_vocab_sizes.keys())
        subseq_max_len = max(subseq_vocab_sizes.keys())
        self.subseq_min_len = subseq_min_len
        self.subseq_max_len = subseq_max_len
        self.subseq_pad_id = subseq_pad_id
        self.subseq_unk_id = subseq_unk_id
        self.subseq_vocab_sizes = subseq_vocab_sizes

        # Create parameters
        self.transition_matrix_z_z = nn.Parameter(torch.randn(Z, Z))
        self.length_emission_matrix_z_n = nn.Parameter(torch.randn(Z, subseq_max_len-subseq_min_len+1))
        # One emission matrix per subsequence length n
        for n in subseq_vocab_sizes:
            subseq_vocab_size = subseq_vocab_sizes[n]
            emission_matrix_subseq_z = nn.Parameter(torch.zeros(subseq_vocab_size, Z))
            setattr(self, f'emission_matrix_subseq_z_{n}gram', emission_matrix_subseq_z)
        self.initialize()
        # Stores normalized emission matrices
        self.normalized_emission_matrices = {}
        # A shared padding that will be used during forward
        self.pad_logits = self.length_emission_matrix_z_n.new(self.Z, self.subseq_min_len).cuda().fill_(NEG_INF)

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def fix_z_transitions(self):
        self.transition_matrix_z_z.requires_grad = False

    def fix_n_emissions(self):
        self.length_emission_matrix_z_n.requires_grad = False

    def forward(self, x, x_lengths, subseq_ids):
        transition_matrix_z_z = self.transition_matrix_z_z.log_softmax(dim=-1)
        length_emission_matrix_z_n = torch.cat((self.pad_logits, self.length_emission_matrix_z_n), dim=-1)
        length_emission_matrix_z_n = length_emission_matrix_z_n.log_softmax(dim=-1) # Z, max_len
        batch_size, seq_len = x.size()
        K = min(self.subseq_max_len+1, seq_len)
        # Prepare scores according to PyTorch Struct's SemiMarkov class
        initial_scores = x.new(self.Z).float().fill_(-float('inf'))
        initial_scores[self.initial_state] = 0
        emission_scores = x.new(batch_size, seq_len, K, self.Z).float().fill_(0)

        for n in range(self.subseq_min_len, K):
            emission_matrix_subseq_z = getattr(self, f'emission_matrix_subseq_z_{n}gram')
            emission_matrix_subseq_z = emission_matrix_subseq_z.log_softmax(dim=0)
            subseq_vocab_size = emission_matrix_subseq_z.size(0)
            subseq_id = subseq_ids[n]
            subseq_id_flat = subseq_id.view(-1)
            subseq_id_flat[subseq_id_flat.ge(subseq_vocab_size)] = self.subseq_unk_id
            emission_scores_n = emission_matrix_subseq_z[subseq_id_flat]
            emission_scores_n.data[subseq_id_flat.eq(self.subseq_unk_id)] = NEG_INF
            emission_scores_n.data[subseq_id_flat.eq(self.subseq_pad_id)] = 0
            emission_scores_n = emission_scores_n.view(subseq_id.shape + (-1,))
            emission_scores[:, :emission_scores_n.size(1), n, :] += emission_scores_n.view(batch_size, -1, self.Z)

        semi = SemiMarkov()
        import pdb; pdb.set_trace()
        logits = semi._dp_standard_efficient(initial_scores, transition_matrix_z_z, length_emission_matrix_z_n, emission_scores, lengths=x_lengths+1)
        return logits


    @torch.no_grad()
    def sample(self, batch_size, M=50):
        self.eval()
        for n in subseq_vocab_sizes:
            if n not in self.normalized_emission_matrices:
                emission_matrix_subseq_z = getattr(self, f'emission_matrix_subseq_z_{n}gram')
                emission_matrix_subseq_z.data[self.subseq_pad_id] = NEG_INF
                emission_matrix_subseq_z.data[self.subseq_unk_id] = NEG_INF
                emission_matrix_subseq_z_normalized = emission_matrix_subseq_z.softmax(dim=0)
                self.normalized_emission_matrices[n] = emission_matrix_subseq_z_normalized

        transition_matrix_z_z_normalized = transition_matrix_z_z.softmax(dim=-1)
        length_emission_matrix_z_n = torch.cat((self.pad_logits, self.length_emission_matrix_z_n), dim=-1)
        length_emission_matrix_z_n_normalized = length_emission_matrix_z_n.softmax(dim=-1) # Z, max_len


        # Sample
        initial_z = torch.new_zeros(batch_size).fill_(self.initial_state)
        z = initial_z
        n_xs = [[] for _ in range(batch_size)]
        for m in range(M):
            probs = transition_matrix_z_z_normalized.gather(0, z.view(-1, 1).expand(-1, self.Z))
            z_sample = torch.distributions.categorical.Categorical(probs).sample().view(batch_size) # batch_size
            length_probs = length_emission_matrix_z_n_normalized.gather(0, z_sample.view(-1, 1).expand(-1, self.subseq_max_len))
            l_sample = torch.distributions.categorical.Categorical(length_probs).sample().view(batch_size) # batch_size
            for i, (z, n) in enumerate(zip(z_sample, length_sample)):
                z = z.item()
                n = n.item()
                emission_matrix_subseq_z_normalized = self.normalized_emission_matrices[n]
                probs = emission_matrix_subseq_z_normalized[:, z]
                x = torch.distributions.categorical.Categorical(probs).sample().view(-1).item()
                n_xs[i].append((n, x))
            z = z_sample
        return n_xs

