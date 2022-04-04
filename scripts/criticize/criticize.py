import sys
import os
import math
import json
import argparse

import torch

parser = argparse.ArgumentParser(description='Fit a critic P_c(z) on the training data.')
parser.add_argument('--critic', type=str, required=True,
                    help='Folder containing critic.pt')
parser.add_argument('--input_file', type=str, required=True,
                    help='Input json file containing predicted_section_names.')
parser.add_argument('--criticised_field', type=str, default='predicted_section_names',
                    help='The field containing section names to be criticised.')
args = parser.parse_args()


def eval_latent_PPL(model, data, criticised_field):
    total_words = 0
    total_loss = 0.
    transition_logits = model['transition_logits']
    stoi = model['vocabulary']
    for sample in data:
        section_names = sample[criticised_field]
        section_names = ['<bos>'] + section_names + ['<eos>']
        for prev_section_name, section_name in zip(section_names[:-1], section_names[1:]):
            if prev_section_name not in stoi or section_name not in stoi:
                print (f'WARNING: invalid section name in transition {prev_section_name} -> {section_name}!')
                continue
            id1 = stoi[prev_section_name]
            id2 = stoi[section_name]
            total_loss += transition_logits[id1, id2]
            total_words += 1

    return math.exp(-total_loss / total_words)


def main(args):
    # Load model
    model = torch.load(os.path.join(args.critic, 'critic.pt'))

    # Criticise text
    data = json.load(open(args.input_file))
    latent_PPL = eval_latent_PPL(model, data, args.criticised_field)
    print (f'Latent PPL: {latent_PPL}')

if __name__ == '__main__':
    main(args)
