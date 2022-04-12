import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_struct import SemiMarkov
from torch_struct.semirings import MaxSemiring

NEG_INF = -1e6

class HSMMModel(nn.Module):

    def __init__(self, device, vocab_size_ngrams, vocab_size_z, max_length_z=11, min_length_z=1, pad_idx=None, train_z=True, train_l=True):
        super().__init__()
        self.mods = {}
        self.device = device
        z_probs_per_z = torch.randn(vocab_size_z, vocab_size_z)#.fill_(0)
        l_probs_per_z = torch.randn(vocab_size_z, max_length_z-min_length_z+1)#.fill_(0)

        #self.z_per_z = nn.Parameter(z_probs_per_z.to(device))
        #self.z_per_z = nn.Parameter(z_probs_per_z).to(device)
        print (f'min length {min_length_z}')
        if train_z:
            print ('train z')
            self.z_per_z = nn.Parameter(z_probs_per_z.to(device))
            #self.l_per_z = nn.Parameter(l_probs_per_z.to(device))
        else:
            print ('not train z')
            self.z_per_z = nn.Parameter(z_probs_per_z).to(device)
            #self.l_per_z = nn.Parameter(l_probs_per_z).to(device)

        if train_l:
            print ('train l')
            #self.z_per_z = nn.Parameter(z_probs_per_z.to(device))
            self.l_per_z = nn.Parameter(l_probs_per_z.to(device))
        else:
            print ('not train l')
            #self.z_per_z = nn.Parameter(z_probs_per_z).to(device)
            self.l_per_z = nn.Parameter(l_probs_per_z).to(device)



        #self.l_per_z = nn.Parameter(l_probs_per_z).to(device)

        self.min_length_z = min_length_z
        self.max_length_z = max_length_z
        self.vocab_size_z = vocab_size_z
        self.vocab_size_ngrams = vocab_size_ngrams
        self.init_state = 0
        assert pad_idx is not None
        self.pad_idx = pad_idx

        self.log_partitions = {}
        for ngram in self.vocab_size_ngrams:
            print (ngram)
            vocab_size_ngram = vocab_size_ngrams[ngram]
            if ngram >= 10:
                MAX_SIZE = 500000
            else:
                MAX_SIZE = 100000
            if vocab_size_z > 300:
                MAX_SIZE = 400000
            else:
                MAX_SIZE = 500000
            if vocab_size_z > 550:
                MAX_SIZE = 300000
            if vocab_size_z > 700:
                MAX_SIZE = 250000
            print ('max_size', MAX_SIZE)
            vocab_size_ngram = min(vocab_size_ngram, MAX_SIZE)
            mod = nn.Parameter(torch.zeros(vocab_size_ngram, vocab_size_z).cuda())#nn.Embedding(vocab_size_ngram, vocab_size_z, sparse=True)
            #mod.data.uniform_(-0.02, 0.02)
            setattr(self, f'emb{ngram}', mod)
        #self.reset_partition()
        self.initialize()

    def initialize(self):
        print ('init with Xavier')
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        #for p in self.parameters():
        #    p.data.uniform_(-0.15, 0.15)

    def forward(self, x, x_lengths, ngram_ids):
        #import pdb; pdb.set_trace()
        #self.factor = self.factor.to(x.device)
        z_per_z = self.z_per_z.clone() # vocab_size_z, vocab_size_z
        
        z_per_z = torch.log_softmax(z_per_z, dim=-1) # vocab_size_z, vocab_size_z
        l_per_z = torch.log_softmax(torch.cat((self.l_per_z.new(self.vocab_size_z, self.min_length_z).fill_(NEG_INF), self.l_per_z), dim=-1), dim=-1) # vocab_size_z, max_length
        #self.ngram_per_z.data[:, :, self.pad_idx] = -float('inf')
        #ngram_per_z = torch.log_softmax(self.ngram_per_z, dim=-1) # vocab_size_z, vocab_size_x ** (ngrams-1), vocab_size_x
        #ngram_per_z.data[:, :, self.pad_idx] = 0
        #ngram_per_z = ngram_per_z.view(self.vocab_size_z, -1)
        # bsz, N_1, K, C, C2 = edge.shape
        # x: bsz, L
        bsz = x.size(0)
        max_length = x.size(1)
        K = min(self.max_length_z+1, max_length)
        #import pdb; pdb.set_trace()
        #ORIG: edge_scores = x.new(bsz, max_length, K, self.vocab_size_z, self.vocab_size_z).float().fill_(0)
        init_z_1 = x.new(self.vocab_size_z).float().fill_(-float('inf'))
        emission_n_l_z = x.new(bsz, max_length, K, self.vocab_size_z).float().fill_(0)
        init_z_1[self.init_state] = 0

        # initial state: 0
        #ORIG: edge_scores[:, 0, :, :, :] = -float('inf') 
        #ORIG: edge_scores[:, 0, :, :, self.init_state] = 0
        # first part: transitions among z's
        #ORIG: edge_scores += z_per_z.view(1, 1, 1, self.vocab_size_z, self.vocab_size_z).transpose(-1, -2)

        # second part: l given z TODO: what about the initial condition?
        #ORIG: edge_scores += l_per_z.transpose(-1, -2).view(1, 1, self.max_length_z+1, self.vocab_size_z, 1)[:, :, :K]

        # last part: x given z and l
        #self.reset_partition()

        self.K = K
        for l in range(self.min_length_z, K):
            #if l not in self.touched_ngram_ids:
            #    self.touched_ngram_ids[l] = set([])
            mod = getattr(self, f'emb{l}')
            #import pdb; pdb.set_trace()
            #mod = mod - mod.logsumexp(dim=0, keepdim=True)
            mod = mod.log_softmax(dim=0)
            #log_partition = self.log_partitions[l]
            #ngram_id = ngram_ids[l].cpu() # bsz, max_length-(l-1)
            ngram_id = ngram_ids[l].cuda() # bsz, max_length-(l-1)
            #if self.training:
            #    self.touched_ngram_ids[l] = self.touched_ngram_ids[l].union(set(ngram_id.view(-1).tolist()))
            #emission_scores = mod(ngram_id) # bsz, max_length-(l-1), vocab_size_z
            ngram_id_flat = ngram_id.view(-1)
            ngram_id_flat[ngram_id_flat.ge(mod.size(0))] = 1 # unk
            emission_scores = mod[ngram_id_flat]
            emission_scores.data[ngram_id_flat.eq(1)] = NEG_INF
            emission_scores.data[ngram_id_flat.eq(0)] = 0
            emission_scores = emission_scores.view(ngram_id.shape + (-1,)) # bsz, max_length-(l-1), vocab_size_z
            #emission_probs = emission_scores - log_partition.view(1, 1, -1) # bsz, max_length - (l-1), vocab_size_z
            emission_probs = emission_scores
            emission_probs = emission_probs.to(self.device)
            #edge_scores[:, :emission_probs.size(1), l, :, :] += emission_probs.view(bsz, -1, self.vocab_size_z, 1)
            emission_n_l_z[:, :emission_probs.size(1), l, :] += emission_probs.view(bsz, -1, self.vocab_size_z)
            ##l = 10
            ##x_unfolded = F.unfold(x.float().unsqueeze(1).unsqueeze(-1), kernel_size=(l, 1), stride=1, padding=(l-1, 0)).squeeze(-1).long().transpose(-1, -2)[:, :-(l-1)]
            #x_unfolded = F.unfold(x.float().unsqueeze(1).unsqueeze(-1), kernel_size=(l, 1), stride=1, padding=(0, 0)).squeeze(-1).long().transpose(-1, -2) # bsz, max_length-(l-1), l
            ## pad with unk's (ngram-1) 
            #paddings = x_unfolded.new_zeros(bsz, x_unfolded.size(1), self.ngrams-1)
            #x_unfolded_padded = torch.cat((paddings, x_unfolded), dim=-1) # bsz, max_length-(l-1), l+ngrams-1
            ## compute ngram scores
            #x_unfolded_padded = x_unfolded_padded.unsqueeze(-1) # bsz, max_length-(l-1), l+ngrams-1, 1
            #x_unfolded_padded_unfolded = F.unfold(x_unfolded_padded.float(), kernel_size=(self.ngrams,1), stride=1, padding=(0,0)).long().view(bsz, max_length-(l-1), self.ngrams, -1).transpose(-1, -2) # bsz, L, l, ngram
            #x_ids = (x_unfolded_padded_unfolded * self.factor.view(1, 1, 1, -1)).sum(-1) # bsz, max_length-(l-1), l
            #emission_probs = ngram_per_z.index_select(1, x_ids.view(-1)).view(self.vocab_size_z, bsz, max_length-(l-1), -1).sum(-1).transpose(0, 1) # bsz, vocab_size_z, max_length-(l-1)
            #edge_scores[:, l-1, :, :, :] += emission_probs[:, :, 0].view(bsz, 1, 1, self.vocab_size_z)
            #edge_scores[:, :emission_probs.size(-1), l, :, :] += emission_probs.transpose(-1, -2).view(bsz, -1, self.vocab_size_z, 1)

        semi = SemiMarkov()
        #import pdb; pdb.set_trace()
        #ORIG: logits_base = semi._dp_standard(edge_scores, x_lengths+1)[0]
        logits = semi._dp_standard_efficient(init_z_1, z_per_z, l_per_z, emission_n_l_z, lengths=x_lengths+1)
        #pass
    #def _dp_standard_efficient(self, init_z_1, transition_z_to_z, transition_z_to_l, emission_n_l_z, lengths=None, force_grad=False):
        return logits

    def argmax(self, x, x_lengths, ngram_ids, dropout=0):
        #import pdb; pdb.set_trace()
        #self.factor = self.factor.to(x.device)
        z_per_z = self.z_per_z.clone() # vocab_size_z, vocab_size_z
        if dropout > 0:
            mask = z_per_z.new(z_per_z.size(1)).bernoulli_(dropout).bool()
            while mask.all():
                mask = z_per_z.new(z_per_z.size(1)).bernoulli_(dropout).bool()
            mask = mask.view(1, -1).expand(self.vocab_size_z, -1)
            z_per_z.data[mask] = NEG_INF
        
        z_per_z = torch.log_softmax(z_per_z, dim=-1) # vocab_size_z, vocab_size_z
        l_per_z = torch.log_softmax(torch.cat((self.l_per_z.new(self.vocab_size_z, self.min_length_z).fill_(NEG_INF), self.l_per_z), dim=-1), dim=-1) # vocab_size_z, max_length
        #self.ngram_per_z.data[:, :, self.pad_idx] = -float('inf')
        #ngram_per_z = torch.log_softmax(self.ngram_per_z, dim=-1) # vocab_size_z, vocab_size_x ** (ngrams-1), vocab_size_x
        #ngram_per_z.data[:, :, self.pad_idx] = 0
        #ngram_per_z = ngram_per_z.view(self.vocab_size_z, -1)
        # bsz, N_1, K, C, C2 = edge.shape
        # x: bsz, L
        bsz = x.size(0)
        max_length = x.size(1)
        K = min(self.max_length_z+1, max_length)
        #import pdb; pdb.set_trace()
        edge_scores = x.new(bsz, max_length, K, self.vocab_size_z, self.vocab_size_z).float().fill_(0)
        #init_z_1 = x.new(self.vocab_size_z).float().fill_(-float('inf'))
        #emission_n_l_z = x.new(bsz, max_length, K, self.vocab_size_z).float().fill_(0)
        #init_z_1[self.init_state] = 0

        # initial state: 0
        edge_scores[:, 0, :, :, :] = -float('inf') 
        edge_scores[:, 0, :, :, self.init_state] = 0
        # first part: transitions among z's
        edge_scores += z_per_z.view(1, 1, 1, self.vocab_size_z, self.vocab_size_z).transpose(-1, -2)

        # second part: l given z TODO: what about the initial condition?
        edge_scores += l_per_z.transpose(-1, -2).view(1, 1, self.max_length_z+1, self.vocab_size_z, 1)[:, :, :K]

        # last part: x given z and l
        #self.reset_partition()

        self.K = K
        for l in range(self.min_length_z, K):
            #if l not in self.touched_ngram_ids:
            #    self.touched_ngram_ids[l] = set([])
            mod = getattr(self, f'emb{l}')
            #import pdb; pdb.set_trace()
            #mod = mod - mod.logsumexp(dim=0, keepdim=True)
            mod = mod.log_softmax(dim=0)
            #log_partition = self.log_partitions[l]
            #ngram_id = ngram_ids[l].cpu() # bsz, max_length-(l-1)
            ngram_id = ngram_ids[l].cuda() # bsz, max_length-(l-1)
            #if self.training:
            #    self.touched_ngram_ids[l] = self.touched_ngram_ids[l].union(set(ngram_id.view(-1).tolist()))
            #emission_scores = mod(ngram_id) # bsz, max_length-(l-1), vocab_size_z
            ngram_id_flat = ngram_id.view(-1)
            ngram_id_flat[ngram_id_flat.ge(mod.size(0))] = 1 # unk
            emission_scores = mod[ngram_id_flat]
            emission_scores.data[ngram_id_flat.eq(1)] = NEG_INF
            emission_scores.data[ngram_id_flat.eq(0)] = 0
            emission_scores = emission_scores.view(ngram_id.shape + (-1,)) # bsz, max_length-(l-1), vocab_size_z
            #emission_probs = emission_scores - log_partition.view(1, 1, -1) # bsz, max_length - (l-1), vocab_size_z
            emission_probs = emission_scores
            emission_probs = emission_probs.to(self.device)
            edge_scores[:, :emission_probs.size(1), l, :, :] += emission_probs.view(bsz, -1, self.vocab_size_z, 1)
            #emission_n_l_z[:, :emission_probs.size(1), l, :] += emission_probs.view(bsz, -1, self.vocab_size_z)
            ##l = 10
            ##x_unfolded = F.unfold(x.float().unsqueeze(1).unsqueeze(-1), kernel_size=(l, 1), stride=1, padding=(l-1, 0)).squeeze(-1).long().transpose(-1, -2)[:, :-(l-1)]
            #x_unfolded = F.unfold(x.float().unsqueeze(1).unsqueeze(-1), kernel_size=(l, 1), stride=1, padding=(0, 0)).squeeze(-1).long().transpose(-1, -2) # bsz, max_length-(l-1), l
            ## pad with unk's (ngram-1) 
            #paddings = x_unfolded.new_zeros(bsz, x_unfolded.size(1), self.ngrams-1)
            #x_unfolded_padded = torch.cat((paddings, x_unfolded), dim=-1) # bsz, max_length-(l-1), l+ngrams-1
            ## compute ngram scores
            #x_unfolded_padded = x_unfolded_padded.unsqueeze(-1) # bsz, max_length-(l-1), l+ngrams-1, 1
            #x_unfolded_padded_unfolded = F.unfold(x_unfolded_padded.float(), kernel_size=(self.ngrams,1), stride=1, padding=(0,0)).long().view(bsz, max_length-(l-1), self.ngrams, -1).transpose(-1, -2) # bsz, L, l, ngram
            #x_ids = (x_unfolded_padded_unfolded * self.factor.view(1, 1, 1, -1)).sum(-1) # bsz, max_length-(l-1), l
            #emission_probs = ngram_per_z.index_select(1, x_ids.view(-1)).view(self.vocab_size_z, bsz, max_length-(l-1), -1).sum(-1).transpose(0, 1) # bsz, vocab_size_z, max_length-(l-1)
            #edge_scores[:, l-1, :, :, :] += emission_probs[:, :, 0].view(bsz, 1, 1, self.vocab_size_z)
            #edge_scores[:, :emission_probs.size(-1), l, :, :] += emission_probs.transpose(-1, -2).view(bsz, -1, self.vocab_size_z, 1)

        semi = SemiMarkov(MaxSemiring)
        #import pdb; pdb.set_trace()
        logits, edges, _ = semi._dp_standard(edge_scores, x_lengths+1)
        obj = MaxSemiring.unconvert(logits).sum(dim=0)
        marg = torch.autograd.grad(
              obj, edges, create_graph=True, only_inputs=True, allow_unused=False
        )
        a_m = marg[0]
        argmax = MaxSemiring.unconvert(a_m)
        #logits = semi._dp_standard_efficient(init_z_1, z_per_z, l_per_z, emission_n_l_z, lengths=x_lengths+1)
        #pass
    #def _dp_standard_efficient(self, init_z_1, transition_z_to_z, transition_z_to_l, emission_n_l_z, lengths=None, force_grad=False):
        return argmax

    @torch.no_grad()
    def get_logsumexps(self):
        logsumexps = {}
        for l in self.touched_ngram_ids:
            touched_ngram_ids = self.touched_ngram_ids[l]
            touched_ngram_ids = torch.tensor(list(touched_ngram_ids)).long().view(-1)
            mod = getattr(self, f'emb{l}')
            scores = mod(touched_ngram_ids) # -1, vocab_size_z
            logsumexps[l] = scores.logsumexp(0)
        return logsumexps

    @torch.no_grad()
    def sample(self, x, num_zs, disable=None):
        self.eval()
        for l in self.vocab_size_ngrams:
            print ('init', l)
            #if l not in self.touched_ngram_ids:
            #    self.touched_ngram_ids[l] = set([])
            if l not in self.mods:
                mod = getattr(self, f'emb{l}')
                mod.data[0] = NEG_INF
                mod.data[1] = NEG_INF
                #mod = mod - mod.logsumexp(dim=0, keepdim=True)
                mod = mod.softmax(dim=0)
                #import pdb; pdb.set_trace()
                self.mods[l] = mod
        #self.factor = self.factor.to(x.device)
        z_init = x.new(x.size(0)).fill_(self.init_state) # bsz
        #import pdb; pdb.set_trace()
        z_per_z = torch.softmax(self.z_per_z, dim=-1) # vocab_size_z, vocab_size_z
        z = z_init
        zs = [[] for _ in x]
        l_per_z = torch.softmax(torch.cat((self.l_per_z.new(self.vocab_size_z, self.min_length_z).fill_(NEG_INF), self.l_per_z), dim=-1), dim=-1) # vocab_size_z, max_length
        #import pdb; pdb.set_trace()
        #ngram_per_z = torch.softmax(self.ngram_per_z, dim=-1) # vocab_size_z, vocab_size_x ** (ngrams-1), vocab_size_x
        #for d in disable:
        #    ngram_per_z[:, :, d] = 0
        #ngram_per_z.data[:, :, self.pad_idx] = 0
        #ngram_per_z = ngram_per_z.view(self.vocab_size_z, -1)
        # first, sample zs
        xs = [[] for _ in x]
        for k in range(num_zs):
            probs = z_per_z.gather(0, z.view(-1,1).expand(-1, self.vocab_size_z)) # bsz, vocab_size_z
            ix = torch.multinomial(probs, num_samples=1).squeeze(-1) # bsz
            l_probs = l_per_z.gather(0, ix.view(-1, 1).expand(-1, l_per_z.size(-1))) # bsz, max_length
            ls = torch.multinomial(l_probs, num_samples=1).squeeze(-1) # bsz
            #ngram_probs = ngram_per_z.gather(0, z.view(-1,1).expand(-1, ngram_per_z.size(-1))) # bsz, vocab_z**ngram
            for i, (z, l) in enumerate(zip(ix, ls)):
                #import pdb; pdb.set_trace()
                l = l.item()
                z = z.item()
                mod = self.mods[l]
                probs = mod[:, z] # probs
                x = torch.multinomial(probs, num_samples=1).view(-1).item()
                xs[i].append((l, x))
                zs[i].append(z)

                #context = x.new_zeros(self.ngrams-1)
                #ngram_prob = ngram_probs[i]
                #for j in range(l):
                #    idx = (context * self.factor[:-1]).sum()
                #    ngram_p = ngram_prob[idx:(idx+self.vocab_size_x)] # vocab_size_x
                #    x = torch.multinomial(ngram_p, num_samples=1).squeeze(-1)
                #    xs[i].append(x.item())
                #    zs[i].append(z.item())
                #    context[:-1] = context[1:]
                #    context[-1] = x.item()
            z = ix
        #import pdb; pdb.set_trace()
        return xs, zs

