import sys
import os
import math
import json
import argparse

import torch
from critic import NGramCritic

parser = argparse.ArgumentParser(description='Fit a critic P_c(z) on the training data.')
parser.add_argument('--train_coreference_chains', type=str, required=True,
                    help='A json file containing coreference chains.')
parser.add_argument('--output_critic_filename', type=str, required=True,
                    help='Output file for the learned critic.')
parser.add_argument('--N', type=int, default=5,
                    help='N in n-gram critic. Default value is 5.')
args = parser.parse_args()


def generate_N_grams(words, ngram, bos='<bos>', eos='<eos>'):
    words = [bos] * (ngram-1) + words + [eos]
    ngrams = list(zip(*[words[i:] for i in range(ngram)]))
    return ngrams


def main(args):
    critic = NGramCritic(N=args.N)
    critic.fit(args.train_coreference_chains)
    critic.save(args.output_critic_filename)
    val_ppl, val_ppls = model.read_val(val_file)

if __name__ == '__main__':
    main(args)
