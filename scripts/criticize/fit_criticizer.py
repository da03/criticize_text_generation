import sys
import os
import math
import json
import argparse

import torch

parser = argparse.ArgumentParser(description='Fit a criticizer P_c(z) on the training data.')
parser.add_argument('--dataset_folder', type=str, required=True,
                    help='Folder containing train.json, val.json, and test.json.')
parser.add_argument('--output_folder', type=str, required=True,
                    help='Output folder for the learned criticizer.')
parser.add_argument('--deltas', type=float, nargs='+', default=[0, 1],
                            help='Smoothing constant values to optimize.')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed.')
args = parser.parse_args()


def eval_latent_PPL(model, data):
    total_words = 0
    total_loss = 0.
    transition_logits = model['transition_logits']
    stoi = model['vocabulary']
    for sample in data:
        section_names = sample['section_names']
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
    # First, establish the vocabulary
    stoi = {}

    train_file = os.path.join(args.dataset_folder, 'train.json')
    val_file = os.path.join(args.dataset_folder, 'val.json')
    test_file = os.path.join(args.dataset_folder, 'test.json')

    data = json.load(open(train_file))
    for sample in data:
        section_names = sample['section_names']
        for section_name in section_names:
            if section_name not in stoi:
                stoi[section_name] = len(stoi)

    stoi['<bos>'] = len(stoi)
    stoi['<eos>'] = len(stoi)

    vocab_size = len(stoi)

    # Next, count bigrams
    bigram_counts = torch.zeros(vocab_size, vocab_size)

    for sample in data:
        section_names = sample['section_names']
        section_names = ['<bos>'] + section_names + ['<eos>']
        for prev_section_name, section_name in zip(section_names[:-1], section_names[1:]):
            id1 = stoi[prev_section_name]
            id2 = stoi[section_name]
            bigram_counts[id1, id2] += 1

    # Next, search for delta that minimizes validation latent PPL
    best_val_latent_PPL = float('inf')
    best_model = None
    best_delta = None
    for delta in args.deltas:
        bigram_counts_smoothed = bigram_counts + delta
        bigram_counts_smoothed /= bigram_counts_smoothed.sum(-1, keepdim=True)
        model = {'transition_logits': bigram_counts_smoothed.log(), 'vocabulary': stoi}
        val_latent_PPL = eval_latent_PPL(model, json.load(open(val_file)))
        if val_latent_PPL < best_val_latent_PPL:
            best_val_latent_PPL = val_latent_PPL
            best_model = model
            best_delta = delta
    print (f'best delta found: {delta}')

    # Save model
    model = best_model
    os.makedirs(args.output_folder, exist_ok=True)
    torch.save(model, os.path.join(args.output_folder, 'criticizer.pt'))


    # Evaluate model
    training_latent_PPL = eval_latent_PPL(model, json.load(open(train_file)))
    val_latent_PPL = best_val_latent_PPL
    test_latent_PPL = eval_latent_PPL(model, json.load(open(test_file)))
    print (f'training latent PPL: {training_latent_PPL}')
    print (f'validation latent PPL: {val_latent_PPL}')
    print (f'test latent PPL: {test_latent_PPL}')

if __name__ == '__main__':
    main(args)
