import sys
import os
import math
import json
import argparse

from critic import NGramCritic

parser = argparse.ArgumentParser(description='Fit a critic P_c(z) on the training data.')
parser.add_argument('--coreference_chains_train', type=str, required=True,
                    help='A json file containing coreference chains.')
parser.add_argument('--output_critic_filename', type=str, required=True,
                    help='Output file for the learned critic.')
parser.add_argument('--N', type=int, default=5,
                    help='N in n-gram critic. Default value is 5.')
args = parser.parse_args()



def main(args):
    critic = NGramCritic(N=args.N)
    critic.fit(args.coreference_chains_train)
    critic.save(args.output_critic_filename)

if __name__ == '__main__':
    main(args)
