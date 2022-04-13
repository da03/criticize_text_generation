import os
import re
import glob
import tqdm

import torch
import torchtext.legacy as tt


class HSMMDataset(tt.data.Dataset):
  @staticmethod
  def sort_key(ex):
    return len(ex.text)

  def __init__(self, path, text_field, ngram_fields, **kwargs):
    fields = [('text', text_field)]
    fields.extend(ngram_fields)
    
    examples = []
    # Get text
    with open(path) as fin:
        for line in fin:
            ex = tt.data.Example()
            ex.text = text_field.preprocess(line.strip()) 
            examples.append(ex)
    
    # Get ngram ids
    for ngram_field_name, ngram_field in tqdm.tqdm(ngram_fields):
        m = re.match(r'(\d+)_gram', ngram_field_name)
        n = int(m.group(1))
        ngram_filename = path + f'.{n}-gram'
        assert os.path.exists(ngram_filename), ngram_filename
        with open(ngram_filename) as fin:
            for i, line in enumerate(fin):
                setattr(examples[i], ngram_field_name, ngram_field.preprocess(line.strip()))
            
    super().__init__(examples, fields, **kwargs)
  
  @classmethod
  def splits(cls, text_field, ngram_fields, path='./',
              train='train', validation='val', test='test',
              **kwargs):
    train_data = None if train is None else cls(
        os.path.join(path, train), text_field, ngram_fields, **kwargs)
    val_data = None if validation is None else cls(
        os.path.join(path, validation), text_field, ngram_fields, **kwargs)
    test_data = None if test is None else cls(
        os.path.join(path, test), text_field, ngram_fields, **kwargs)
    return tuple(d for d in (train_data, val_data, test_data)
                   if d is not None)


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


def load_data(dataset_folder, vocab_folder, Z, batch_size, pad_token, unk_token):
    # Load subseq vocabs
    vocabs = load_subseq_vocabs(vocab_folder)
    subseq_min_len = min(vocabs.keys())
    subseq_max_len = max(vocabs.keys())
    subseq_vocab_sizes = {n: len(vocabs[n]) for n in vocabs}

    # Build fields
    def tokenize(s):
        return [int(item) for item in s.split()]

    ngram_fields = []
    for n in range(subseq_min_len, subseq_max_len+1):
        vocab = vocabs[n]
        subseq_pad_id = vocab[pad_token]
        subseq_unk_id = vocab[unk_token]
        ngram_field = tt.data.Field(include_lengths=True,
                batch_first=True,
                tokenize=tokenize,
                use_vocab=False,
                pad_token=vocab[pad_token],
                )
        ngram_fields.append((f'{n}_gram', ngram_field))

    text_field = tt.data.Field(include_lengths=True,
            batch_first=True,
            tokenize=lambda x: x.split(),
            )

    # Build dataset splits
    train_data, val_data, test_data = HSMMDataset.splits(
            text_field, ngram_fields, path=dataset_folder,
            train='train.x', validation='val.x', test='test.x')

    # Build vocabulary for text field
    text_field.build_vocab(train_data.text)
    x_pad_id = text_field.vocab.stoi[text_field.pad_token]

    x_vocab_size = len(text_field.vocab)
    print (f"Size of text vocab: {x_vocab_size}")

    # Create data iterators
    train_iter, val_iter, test_iter= tt.data.BucketIterator.splits((train_data, val_data, test_data),
            batch_size=batch_size, 
            device=torch.device('cuda'),
            repeat=False, 
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            )

    return train_iter, val_iter, test_iter, subseq_vocab_sizes, x_pad_id, subseq_pad_id, subseq_unk_id
