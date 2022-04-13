import torch
import sys
import re
import argparse
import tqdm

from data import load_subseq_vocabs
from model import HSMMModel

parser = argparse.ArgumentParser(description='Sample from a trained HSMM language model.')
parser.add_argument('--checkpoint_path', type=str, default='language_model_checkpoints/hsmm/checkpoint_best.pt',
                    help='The path of the trained model.')
parser.add_argument('--output_file', type=str, required=True,
                    help='Output file storing samples from the language model.')
parser.add_argument('--vocab_folder', type=str, default='language_model_checkpoints/hsmm',
                    help='Vocabulary folder storing the constructed vocabulary files subseq_vocab_n-gram.txt.')
parser.add_argument('--num_samples', type=int, default=6400,
                    help='Number of samples to generate. Should be a multiple of batch_size.')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for sampling.')
parser.add_argument('--M', type=int, default=50,
                    help='Number of states in each sequence.')
parser.add_argument('--pad_token', type=str, default='<pad>',
                    help='Special token reserved for padding.')
parser.add_argument('--unk_token', type=str, default='<unk>',
                    help='Special token reserved for unknown subsequences.')

def main(args):
    # Load subseq vocabs
    subseq_vocabs, subseq_vocabs_itos = load_subseq_vocabs(args.vocab_folder, get_itos=True)
    subseq_vocab_sizes = {n: len(subseq_vocabs[n]) for n in subseq_vocabs}
    for n in subseq_vocabs:
        subseq_pad_id = subseq_vocabs[n][args.pad_token]
        subseq_unk_id = subseq_vocabs[n][args.unk_token]
        break

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))

    # Find tensor sizes
    transition_matrix_z_z = checkpoint['transition_matrix_z_z']
    Z = transition_matrix_z_z.size(0)

    # Build model
    model = HSMMModel(Z, subseq_vocab_sizes, subseq_pad_id=subseq_pad_id, subseq_unk_id=subseq_unk_id)
    model.load_state_dict(checkpoint) 

    with open(args.output_file, 'w') as fout:
        num_batches = args.num_samples // args.batch_size
        for _ in tqdm.tqdm(range(num_batches)):
            n_xs = model.sample(args.batch_size, args.M)
            for n_x in n_xs:
                subseqs = []
                for n, x in n_x:
                    subseqs.append(subseq_vocabs_itos[n][x])
                x_str = ' '.join(subseqs)
                fout.write(x_str + '\n')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
