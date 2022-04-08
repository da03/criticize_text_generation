import sys
import os
import math
import json
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch
import seaborn as sns

# usage: python topics.py <beta-file> <vocab-file> <num words>
parser = argparse.ArgumentParser(description='Compute Latent NLL.')
parser.add_argument('--critic', type=str, required=True,
                    help='Folder containing critic checkpoints')
parser.add_argument('--vocab_file', type=str, required=True,
                    help='Vocabulary file generated by scripts/data/build_data_for_critics.py. Should be data_folder/train.json.CTM.vocab.')
parser.add_argument('--top_k', type=int, default=10,
                    help='Only print out the top top_k words per topic.')
args = parser.parse_args()

def load_matrix(filename, num_cols):
    with open(filename) as fin:
        nums = [float(line.strip()) for line in fin.readlines()]
        array = torch.Tensor(nums).view(-1, num_cols)
    return array

def load_params(param_filename):
    params = {}
    with open(param_filename) as fin:
        for line in fin:
            key, val = line.strip().split()
            params[key] = val
    return params

def load_vocabulary(vocab_filename):
    itos = []
    with open(vocab_filename) as fin:
        for line in fin:
            itos.append(line.strip())
    return itos

def main(args):
    # Load model
    params = load_params(os.path.join(args.critic, 'final-param.txt'))
    num_terms = int(params['num_terms'])

    itos = load_vocabulary(args.vocab_file)

    topic_word_file = os.path.join(args.critic, 'final-log-beta.dat')
    topic_word_matrix = load_matrix(topic_word_file, num_terms)

    top_word_ids = topic_word_matrix.argsort(-1, descending=True)

    for i in range(top_word_ids.size(0)):
        print(f'topic {i}:')
        for j in range(args.top_k):
            print (f'\t {itos[top_word_ids[i][j].item()]}', end='')
        print ('\n')


if __name__ == '__main__':
    main(args)