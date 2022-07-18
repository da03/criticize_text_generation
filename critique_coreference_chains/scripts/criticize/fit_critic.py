import sys
import os
import math
import json
import argparse

import torch

parser = argparse.ArgumentParser(description='Fit a critic P_c(z) on the training data.')
parser.add_argument('--train_coreference_chains', type=str, required=True,
                    help='A json file containing coreference chains.')
parser.add_argument('--output_critic_filename', type=str, required=True,
                    help='Output file for the learned critic.')
args = parser.parse_args()


def main(args):

if __name__ == '__main__':
    main(args)
