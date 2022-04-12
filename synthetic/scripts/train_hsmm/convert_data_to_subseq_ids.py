import os
import re
import glob
import argparse
import tqdm

parser = argparse.ArgumentParser(description='Convert ngrams in the datasets into subsequence ids.')
parser.add_argument('--dataset_folder', type=str, required=True,
                    help='Data folder containing train.x.')
parser.add_argument('--vocab_folder', type=str, default='language_model_checkpoints/hsmm',
                    help='Vocabulary folder storing the constructed vocabulary files subseq_vocab_n-gram.txt.')
parser.add_argument('--pad_token', type=str, default='<pad>',
                    help='Special token reserved for padding.')
parser.add_argument('--unk_token', type=str, default='<unk>',
                    help='Special token reserved for unknown subsequences.')
args = parser.parse_args()


def load_subseq_vocabs(foldername):
    filenames = glob.glob(os.path.join(foldername, 'subseq_vocab_*-gram.txt'))
    vocabs = {}
    for filename in filenames:
        m = re.match(r'subseq_vocab_(\d+)-gram.txt', os.path.basename(filename))
        if m:
            n = int(m.group(1))
            vocab = {}
            with open(filename) as fin:
                for line in fin:
                    vocab[line.strip()] = len(vocab)
            vocabs[n] = vocab
    return vocabs

def process_file(input_filename, subseq_vocabs):
    min_n = min(subseq_vocabs.keys())
    max_n = max(subseq_vocabs.keys())
    for n in tqdm.tqdm(range(min_n, max_n+1)):
        vocab = subseq_vocabs[n]
        output_filename = input_filename + f'.{n}-gram'
        with open(input_filename) as fin:
            with open(output_filename, 'w') as fout:
                for line in fin:
                    tokens = line.strip().split()
                    ngram_ids = []
                    for i in range(len(tokens)):
                        if i+n <= len(tokens):
                            subseq = ' '.join(tokens[i:(i+n)])
                            if subseq not in vocab:
                                subseq = args.unk_token
                            ngram_id = vocab[subseq]
                            ngram_ids.append(ngram_id)
                    fout.write(' '.join([str(ngram_id) for ngram_id in ngram_ids]) + '\n')


def main(args):
    dataset_folder = args.dataset_folder
    vocab_folder = args.vocab_folder
    pad_token = args.pad_token
    unk_token = args.unk_token

    subseq_vocabs = load_subseq_vocabs(args.vocab_folder)

    for split in ['train', 'val', 'test']:
        input_filename = os.path.join(dataset_folder, f'{split}.x')
        print (f'Processing {input_filename}')
        process_file(input_filename, subseq_vocabs)


if __name__ == '__main__':
    main(args)
