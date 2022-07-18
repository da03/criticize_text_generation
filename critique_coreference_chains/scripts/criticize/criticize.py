import sys
import os
import math
import json
import argparse

import pickle
from critic import NGramCritic

parser = argparse.ArgumentParser(description='Compute Latent PPL.')
parser.add_argument('--critic', type=str, required=True,
                    help='Folder containing critic.pt')
parser.add_argument('--coreference_chains', type=str, required=True,
                    help='Input json file containing coreference chains.')
args = parser.parse_args()


def main(args):
    # Load critic
    critic = pickle.load(open(args.critic, 'rb'))

    # Criticise text
    latent_PPL = critic.evaluate_latent_PPL(args.coreference_chains)

    print (f'Latent PPL: {latent_PPL}')

if __name__ == '__main__':
    main(args)
