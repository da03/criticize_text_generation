import sys
import os
import math
import torch
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--sample_file", type=str, required=True)
parser.add_argument("--word2letters_file", type=str, required=True)
parser.add_argument("--id2clusters_file", type=str, required=True)
parser.add_argument("--transition_file", type=str, required=True)
parser.add_argument("--emission_file", type=str, required=True)
parser.add_argument("--initial_state", type=int, default=0)
#parser.add_argument("--unk_x", type=int, default=0)
parser.add_argument("--unk_word", type=int, default=0)
parser.add_argument("--max_words", type=int, default=100)
#parser.add_argument("--unk_x_emission_prob", type=float, default=1e-4)

def main(args):
    ignored = 0
    ignored_not = 0
    total_words = 0
    warning_words = 0
    word2letters = []
    letters2word = {}
    #import pdb; pdb.set_trace()
    with open(args.word2letters_file) as fin:
        for idx, line in enumerate(fin):
            word = line.strip().replace(' ', '').replace('<space>', '')
            word2letters.append(word)
            letters2word[word] = idx
    unk_word = word2letters[args.unk_word]

    #import pdb; pdb.set_trace()
    transitions = torch.load(args.transition_file)
    emissions = torch.load(args.emission_file)

    id2clusters = []
    with open(args.id2clusters_file) as fin:
        for line in fin:
            id2clusters.append(int(line.strip()))

    with open(args.sample_file) as fin:
        logprob_z_total = 0.
        logprob_x_given_z_total = 0.
        normalizer_z = 0
        normalizer_x = 0
        for line in fin:
            #normalizer_x += len(line.strip().split())
            initial_state = args.initial_state
            line = line.strip().replace(' ', '')
            words = line.split('<space>')
            words = [item for item in words if item != '']
            #xs = []
            #zs = []
            z = args.initial_state
            #normalizer_z += len(words)
            #import pdb; pdb.set_trace()
            logprob_z_total_example = 0.
            logprob_x_given_z_total_example = 0.
            normalizer_z_example = 0.
            normalizer_x_example = 0.
            warning = False
            iiii = 0
            for word in words:
                iiii += 1
                if iiii > args.max_words:
                    break
                #import pdb; pdb.set_trace()
                if word not in letters2word:
                    #import pdb; pdb.set_trace()
                    word = unk_word
                    warning_words += 1
                    warning = True
                    print ('WARNING')
                total_words += 1
                x = letters2word[word]
                new_z = id2clusters[x]
                #import pdb; pdb.set_trace()
                #import random
                #new_z = random.choice(list(range(256)))
                logprob_z = transitions[z][new_z].log()
                z = new_z
                logprob_x_given_z = emissions[z][x].log()
                if word != unk_word:
                  normalizer_z += 1
                  normalizer_x += (1 + len(word))
                  logprob_z_total += logprob_z
                  logprob_x_given_z_total += logprob_x_given_z
                  normalizer_z_example += 1
                  normalizer_x_example += (1 + len(word))
                  logprob_z_total_example += logprob_z
                  logprob_x_given_z_total_example += logprob_x_given_z
                #else:
                #  normalizer_z -= 1
                #  normalizer_x -= (1+len(word))
                #xs.append(x)
                #zs.append(z)
            ignored_not += 1
            if warning:
              ignored += 1
              normalizer_z -= normalizer_z_example
              normalizer_x -= normalizer_x_example
              logprob_z_total -= logprob_z_total_example
              logprob_x_given_z_total -= logprob_x_given_z_total_example
        print (f'warning ratio: {warning_words/total_words}')
        print (f'z PPL: {math.exp(-logprob_z_total / normalizer_z)}, x given z PPL: {math.exp(-logprob_x_given_z_total / normalizer_x)}, x shortest given z PPL:  {math.exp(-logprob_x_given_z_total / normalizer_z)}, flat PPL: {math.exp(-(logprob_z_total+logprob_x_given_z_total) / normalizer_x)}, flat shortest PPL: {math.exp(-(logprob_z_total+logprob_x_given_z_total) / normalizer_z)}')
        print (ignored/ignored_not)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
