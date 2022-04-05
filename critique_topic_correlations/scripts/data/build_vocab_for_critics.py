import os
import json
import argparse
import random
import collections


parser = argparse.ArgumentParser(description='Build vocabulary which will be later used for processing data for fitting critic generative processes.')
parser.add_argument('--dataset_folder', type=str, required=True,
                    help='Folder containing train.json, val.json, and test.json.')
parser.add_argument('--compatible_with_checkpoints', action='store_true',
                    help='In the pretrained models, the last document in the wiki dataset was skipped due to an implementation error. This flag needs to be set to use the pretrained checkpoints.')
parser.add_argument('--min_term_freq', type=int, default=5,
                    help='Only include a word type in the vocabulary if its frequency is greater than or equal to this value.')
parser.add_argument('--max_doc_freq_ratio', type=float, default=0.5,
                    help='Only include a word type in the vocabulary if it appears in less than this value proportion of documents.')
parser.add_argument('--ignore_containing', type=str, nargs='+', default=['xmath',],
                            help='Ignore words containing these strings.')
parser.add_argument('--stopwords_file', type=str, default='scripts/data/stopwords.txt',
                    help='File containing stopwords (one per line).')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed.')
args = parser.parse_args()


def compute_word_doc_freqs(filename):
    word_df = collections.defaultdict(int)
    data = json.load(open(filename))
    num_docs = len(data)
    for example in data:
        sections = example['sections']
        words = ' '.join(sections)
        words = words.replace('\n', ' ')
        words = words.split()
        words = set(words)
        for word in words:
            word_df[word] += 1
    for word in word_df:
        word_df[word] = word_df[word] / num_docs
    return word_df


def load_stopwords(stopwords_file):
    stopwords = set([])
    with open(stopwords_file) as fin:
        for line in fin:
            stopwords.add(line.strip().lower())
    return stopwords


def ignore_word(word, stopwords, word_df, ignore_containing, max_doc_freq_ratio):
    if word.lower() in stopwords:
        return True
    if word_df[word] >= max_doc_freq_ratio:
        return True
    for s in ignore_containing:
        if s.lower() in word.lower():
            return True
    return False


def construct_vocab(filename, filename_out, stopwords, word_df, ignore_containing, min_term_freq, max_doc_freq_ratio, compatible_mode=False):
    freqs = collections.defaultdict(int)
    data = json.load(open(filename))
    if compatible_mode:
        exclude = 101995 # this is originally the last document, but due to shuffling it appears in the middle
        data = data[:exclude] + data[exclude+1:] # drop a document in the compatible mode due to an implementation error that ignored the last document
    for example in data:
        sections = example['sections']
        words = ' '.join(sections)
        words = words.replace('\n', ' ')
        words = words.split()
        for word in words:
            if ignore_word(word, stopwords, word_df, ignore_containing, max_doc_freq_ratio):
                continue
            freqs[word] += 1
    word_freqs = [(word, freqs[word]) for word in freqs]
    word_freqs = sorted(word_freqs, key=lambda x: -x[1])
    itos = []
    for word, freq in word_freqs:
        if freq < min_term_freq:
            break
        itos.append(word)
    return itos


def main(args):
    dataset_folder = args.dataset_folder
    train_file = os.path.join(dataset_folder, 'train.json')

    word_df = compute_word_doc_freqs(train_file)
    stopwords = load_stopwords(args.stopwords_file)
    ignore_containing = args.ignore_containing

    vocab_file = train_file + '.CTM.vocab'
    compatible_mode = False
    if args.compatible_with_checkpoints and 'wiki' in dataset_folder.lower():
        compatible_mode = True
    vocab = construct_vocab(train_file, vocab_file, stopwords, word_df, ignore_containing, args.min_term_freq, args.max_doc_freq_ratio, compatible_mode)
    with open(vocab_file, 'w') as fout:
        for word in vocab:
            fout.write(word + '\n')


if __name__ == '__main__':
    main(args)
