import os
import random
import string
import math
import json
import argparse
import collections
import tqdm

import torch

parser = argparse.ArgumentParser(description='Generate synthetic dataset according to an HSMM.')
parser.add_argument('--dataset_folder', type=str, required=True,
                    help='Output folder storing the generated data.')
parser.add_argument('--M', type=int, default=50,
                    help='Each example will contain a sequence of M states.')
parser.add_argument('--Z', type=int, default=256,
                    help='Each state will take one out of Z possible values.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='The batch size for generating data.')
parser.add_argument('--num_batches', type=int, default=1000,
                    help='The total number of batches to generate.')
parser.add_argument('--train_split_ratio', type=float, default=0.8,
                    help='The ratio of training examples.')
parser.add_argument('--val_split_ratio', type=float, default=0.1,
                    help='The ratio of test examples.')
parser.add_argument('--test_split_ratio', type=float, default=0.1,
                    help='The ratio of test examples.')
parser.add_argument('--subseq_min_len', type=int, default=4,
                    help='Each state will generate at least subseq_min_len words.')
parser.add_argument('--subseq_max_len', type=int, default=11,
                    help='Each state will generate at most subseq_min_len words.')
parser.add_argument('--subseq_vocab_size', type=int, default=10000,
                    help='Number of possible subsequences.')
parser.add_argument('--emission_temperature', type=float, default=0.3,
                    help='Emission logits are first sampled from a Gaussian, and then divided by this value.')
parser.add_argument('--transition_temperature', type=float, default=0.5,
                    help='Transition logits are first sampled from a Gaussian, and then divided by this value.')
parser.add_argument('--initial_state', type=int, default=0,
                    help='The initial (auxiliary) state z_0. We only use it for sampling z_1 based on the transition matrix, but not for generating words.')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed.')
parser.add_argument('--delimiter', type=str, default='<space>',
                    help='Delimiter to mark the boundaries of subsequences.')
args = parser.parse_args()


def main(args):
    dataset_folder = args.dataset_folder
    Z = args.Z
    M = args.M
    batch_size = args.batch_size
    num_batches = args.num_batches
    train_split_ratio = args.train_split_ratio
    val_split_ratio = args.val_split_ratio
    test_split_ratio = args.test_split_ratio
    assert train_split_ratio + val_split_ratio + test_split_ratio == 1
    subseq_min_len = args.subseq_min_len
    subseq_max_len = args.subseq_max_len
    subseq_vocab_size = args.subseq_vocab_size
    emission_temperature = args.emission_temperature
    transition_temperature = args.transition_temperature
    initial_state = args.initial_state

    os.makedirs(dataset_folder, exist_ok=True)
    print (f'dataset folder: {dataset_folder}, Z: {Z}, M: {M}')
    print (f'emission temperature: {emission_temperature}, transition temperature: {transition_temperature}')

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build vocabulary (of words)
    # Vocabulary is 'a' to 'z', 'A' to 'Z', and delimiter
    delimiter = args.delimiter
    itos = [delimiter]
    stoi = {delimiter: 0}
    letters = string.ascii_letters
    for c in letters:
        itos.append(c)
        stoi[c] = len(stoi)

    # Build vocabulary (of subseqs)
    subseq_vocab = []
    subseq_vocab_set = set([])
    for word in range(subseq_vocab_size):
        while True:
            # First, sample length from uniform(subseq_min_len, subseq_max_len)
            l = random.choice(list(range(subseq_min_len, subseq_max_len+1)))
            # Subtract 1 since the last token of each subseq is always delimiter
            l -= 1
            s = ' '.join(random.choice(letters) for i in range(l)) + ' ' + delimiter
            if s not in subseq_vocab_set:
                break
        subseq_vocab_set.add(s)
        subseq_vocab.append(s)
    # Write vocabulary of subseqs to disk
    with open(f'{dataset_folder}/subseq_vocab.txt', 'w') as fout:
        for s in subseq_vocab:
            fout.write(f'{s}\n')

    # For each sub-sequence select a random unique latent state
    id2cluster = torch.randint(0, Z, (subseq_vocab_size, ))
    cluster2id = id2cluster.new(Z, subseq_vocab_size).fill_(0)
    cluster2id.scatter_add_(0, id2cluster.view(1, -1), id2cluster.new_ones(id2cluster.size()).view(1, -1))
    cluster2id = cluster2id.gt(0)

    # Initialize transition matrix
    transition_matrix = torch.randn(Z, Z) / transition_temperature
    transition_matrix = transition_matrix.softmax(dim=-1)

    # Initialize emission matrix
    emission_matrix = torch.randn(Z, subseq_vocab_size) / emission_temperature
    # Mask emission matrix such that each sub-sequence comes from a unique latent state
    emission_matrix[~cluster2id] = -float('inf')
    emission_matrix = emission_matrix.softmax(dim=-1)

    # Write transition and emission to disk
    torch.save(transition_matrix, f'{dataset_folder}/transitions.pt')
    torch.save(emission_matrix, f'{dataset_folder}/emissions.pt')
    with open(f'{dataset_folder}/subseqid_to_z.txt', 'w') as fout:
        for item in id2cluster.view(-1).tolist():
            fout.write(f'{item}\n')

    # Compute Latent PPL (Data) using dynamic programming
    Ep_logp_total_z = 0.
    probs_z = torch.zeros(Z)
    probs_z[initial_state] = 1
    for t in range(M):
        p_logp = transition_matrix * transition_matrix.log()
        Ep_logp_total_z += (probs_z.view(-1, 1) * p_logp).sum()
        probs_z = probs_z @ transition_matrix
        probs_z = probs_z.view(-1)
    latent_PPL = math.exp(-Ep_logp_total_z / M)
    print (f'Latent PPL (data) computed using DP: {latent_PPL}')

    torch.manual_seed(args.seed)
    data = []
    for batch_id in tqdm.tqdm(range(num_batches)):
        state = torch.LongTensor([initial_state,]*batch_size)
        states = torch.zeros(M, batch_size).long()
        words = torch.zeros(M, batch_size).long()
        for t in range(M):
            probs = transition_matrix.gather(0, state.view(-1, 1).expand(-1, Z)) # batch_size, Z
            state = torch.distributions.categorical.Categorical(probs).sample().view(batch_size) # batch_size
            states[t] = state
            emission_probs = emission_matrix.gather(0, state.view(-1, 1).expand(-1, subseq_vocab_size)) # batch_size, subseq_vocab_size
            word = torch.distributions.categorical.Categorical(emission_probs).sample().view(batch_size)
            words[t] = word
        for state, word in zip(states.T, words.T):
            data.append((state, word))

    train_size = int(math.floor(len(data) * train_split_ratio))
    val_size = int(math.floor(len(data) * val_split_ratio))
    train_data = data[:train_size]
    val_data = data[train_size:(train_size+val_size)]
    test_data = data[(train_size+val_size):]

    # Compute PPL of the true generative process on test set
    # This establishes an upper bound on language modeling performance
    total_NLL = 0
    total_words = 0
    for state, word in test_data:
        for s1, s2 in zip([initial_state,]+state[:-1].tolist(), state.tolist()):
            total_NLL += -transition_matrix[s1, s2].log().item()
        for s, w in zip(state, word):
            total_NLL += -emission_matrix[s,w].log().item()
            total_words += len(subseq_vocab[w].strip().split())
        PPL = math.exp(total_NLL / total_words)
    print (f'PPL of the true generative process: {PPL}')

    # Write data to disk
    def write_data(data, split):
        with open(f'{dataset_folder}/{split}.z', 'w') as fz:
            with open(f'{dataset_folder}/{split}.x', 'w') as fx:
                for state, word in data:
                    fz.write(' '.join([str(item) for item in state.tolist()]) + '\n')
                    fx.write(' '.join([subseq_vocab[item] for item in word.tolist()]) + '\n')
    write_data(train_data, 'train')
    write_data(val_data, 'val')
    write_data(test_data, 'test')
    
if __name__ == '__main__':
    main(args)
