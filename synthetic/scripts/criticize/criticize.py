import sys
import os
import math
import json
import argparse

import torch

parser = argparse.ArgumentParser(description='Compute Latent PPL.')
parser.add_argument('--dataset_folder', type=str, required=True,
                    help='Data folder containing transitions.pt which will be used for evaluating P_c(z).')
parser.add_argument('--input_file_z', type=str, required=True,
                    help='Input filename. Each line should be a sequence of z\'s (not the original x\'s).')
parser.add_argument('--initial_state', type=int, default=0,
                    help='The initial (auxiliary) state z_0.')
args = parser.parse_args()


def eval_latent_PPL(transitions, filename, initial_state):
    total_words = 0
    total_loss = 0.
    with open(filename) as fin:
        for line in fin:
            zs = line.strip().split()
            zs = [int(z) for z in zs]
            zs = [initial_state,] + zs
            for z_t_minus_one, z_t in zip(zs[:-1], zs[1:]):
                total_loss += transitions[z_t_minus_one, z_t].log().item()
                total_words += 1

    return math.exp(-total_loss / total_words)


def main(args):
    dataset_folder = args.dataset_folder

    # Load model
    transition_file = os.path.join(dataset_folder, 'transitions.pt')
    transitions = torch.load(transition_file)

    # Criticise real data
    input_filename_z = os.path.join(dataset_folder, 'test.z')
    latent_PPL_real = eval_latent_PPL(transitions, input_filename_z, args.initial_state)

    # Criticise model generations
    input_filename_z = args.input_file_z
    latent_PPL_model = eval_latent_PPL(transitions, input_filename_z, args.initial_state)
    print (f'Latent PPL (data): {latent_PPL_real}. Latent PPL (model): {latent_PPL_model}')

if __name__ == '__main__':
    main(args)
