import os
import random
import string
import math
import json
import argparse
import collections
import tqdm

import torch

parser = argparse.ArgumentParser(description='Build the vocabulary of possible subsequences by finding the most common n-grams in the training data.')
parser.add_argument('--dataset_folder', type=str, required=True,
                    help='Data folder containing train.x.')
parser.add_argument('--output_folder', type=str, default='language_model_checkpoints/hsmm',
                    help='Output folder storing the constructed vocabulary.')
parser.add_argument('--subseq_min_len', type=int, default=1,
                    help='Minimum number of tokens of each subsequence. This will determine the lower bound of n when counting n-grams.')
parser.add_argument('--subseq_max_len', type=int, default=11,
                    help='Maximum number of tokens of each subsequence. This will determine the upper bound of n when counting n-grams.')
parser.add_argument('--subseq_vocab_size', type=int, default=250000,
                    help='Number of possible subsequences per length. For any n, the most common this number of n-grams will be used as subseq vocabulary.')
parser.add_argument('--pad_token', type=str, default='<pad>',
                    help='Special token reserved for padding.')
parser.add_argument('--unk_token', type=str, default='<unk>',
                    help='Special token reserved for unknown subsequences.')
args = parser.parse_args()


def count_ngrams(n, filename, topk=-1):
    ngram_counts = collections.defaultdict(int)
    with open(filename) as fin:
        for line in fin:
            tokens = line.strip().split()
            for i in range(len(tokens)):
                if i+n <= len(tokens):
                    subseq = ' '.join(tokens[i:(i+n)])
                    ngram_counts[subseq] += 1
    ngrams= [(subseq, ngram_counts[subseq]) for subseq in ngram_counts]
    ngrams = sorted(ngrams, key=lambda x: (-x[1], x[0]))
    # Only keep the topk ngrams
    if topk > 0:
        ngrams = ngrams[:topk]
    ngrams = [ngram[0] for ngram in ngrams]
    return ngrams


def main(args):
    dataset_folder = args.dataset_folder
    output_folder = args.output_folder
    subseq_min_len = args.subseq_min_len
    subseq_max_len = args.subseq_max_len
    subseq_vocab_size = args.subseq_vocab_size
    pad_token = args.pad_token
    unk_token = args.unk_token

    train_filename = os.path.join(dataset_folder, 'train.x')

    # Count the most common `subseq_vocab_size` ngrams for n in [subseq_min_len, subseq_max_len]
    ngrams = {}
    for n in tqdm.tqdm(range(subseq_min_len, subseq_max_len+1)):
        ngrams[n] = count_ngrams(n, train_filename, topk=subseq_vocab_size)

    # Dump to disk
    os.makedirs(output_folder, exist_ok=True)
    for n in range(subseq_min_len, subseq_max_len+1):
        output_filename = os.path.join(output_folder, f'subseq_vocab_{n}-gram.txt')
        with open(output_filename, 'w') as fout:
            fout.write(pad_token + '\n')
            fout.write(unk_token + '\n')
            for subseq in ngrams[n]:
                fout.write(subseq + '\n')

        print (f'Vocab written to {output_filename}.')
   

if __name__ == '__main__':
    main(args)
