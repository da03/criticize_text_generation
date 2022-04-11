import os
import random
import string
import math
import json
import argparse
import collections
import tqdm

import torch

parser = argparse.ArgumentParser(description='Infer latent states z from x. The mapping is deterministic.')
parser.add_argument('--dataset_folder', type=str, required=True,
                    help='Data folder containing subseq_vocab.txt and subseqid_to_z.txt which are necessary for posterior inference.')
parser.add_argument('--input_file', type=str, required=True,
                    help='Input filename. Each line should be a generated sequence.')
parser.add_argument('--output_file', type=str, required=True,
                    help='Output filename. Each line should be a predicted sequence of z\'s.')
parser.add_argument('--delimiter', type=str, default='<space>',
                    help='Delimiter to mark the boundaries of subsequences.')
args = parser.parse_args()


def load_subseq_vocabulary(filename):
    stoi = {}
    with open(filename) as fin:
        for line in fin:
            subseq = line.strip()
            stoi[subseq] = len(stoi)
    return stoi


def load_subseqid_to_z(filename):
    itoi = []
    with open(filename) as fin:
        for line in fin:
            z = int(line.strip())
            itoi.append(z)
    return itoi

def find_M(filename):
    with open(filename) as fin:
        line = fin.readline()
        return len(line.strip().split())

def main(args):
    dataset_folder = args.dataset_folder

    # Load the vocabulary of subsequences
    subseq_vocab_filename = os.path.join(dataset_folder, 'subseq_vocab.txt')
    subseq_stoi = load_subseq_vocabulary(subseq_vocab_filename)

    # Load the mapping from subsequence id to z
    subseqid_to_z_filename = os.path.join(dataset_folder, 'subseqid_to_z.txt')
    subseqid_to_z = load_subseqid_to_z(subseqid_to_z_filename)

    # Find the ground truth number of states M
    train_z_filename = os.path.join(dataset_folder, 'train.z')
    M = find_M(train_z_filename)

    # Perform posterior inference
    invalid = 0
    total = 0
    with open(args.input_file) as fin:
        with open(args.output_file, 'w') as fout:
            for line in fin:
                subseqs = line.strip().split(args.delimiter)
                zs = []
                flag_invalid = False
                for subseq in subseqs[:-1]:
                    subseq = subseq.strip() + ' ' + args.delimiter
                    if subseq not in subseq_stoi:
                        flag_invalid = True
                        break
                    subseq_id = subseq_stoi[subseq]
                    z = subseqid_to_z[subseq_id]
                    zs.append(z)
                if len(zs) != M:
                    flag_invalid = True
                    continue
                if flag_invalid:
                    invalid += 1
                total += 1
                fout.write(' '.join([str(z) for z in zs]) + '\n')

    print (f'{total} sequences processed. {invalid} invalid.')
   

if __name__ == '__main__':
    main(args)
