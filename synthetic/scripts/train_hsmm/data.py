import math
import warnings
import copy
import os

import torch
import torch.nn as nn
import torchtext.legacy as tt
import matplotlib.pyplot as plt

from tqdm import tqdm

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


NGRAMS_MIN = 1
NGRAMS_MAX = 11

data_folder = 'hsmm_data'
BATCH_SIZE = 4     # batch size for training and validation
BATCH_SIZE = 8     # batch size for training and validation
print (data_folder)


class CatDataset(tt.data.Dataset):
  @staticmethod
  def sort_key(ex):
    return len(ex.text)

  def __init__(self, path, text_field, ngram_fields, **kwargs):
    fields = [('text', text_field)]
    fields.extend(ngram_fields)
    
    examples = []
    # Get text
    with open(path+'.x', 'r') as f:
        for line in f:
            ex = tt.data.Example()
            ex.text = text_field.preprocess(line.strip()) 
            examples.append(ex)
    
    # Get ngram ids
    for ngram, ngram_field in ngram_fields:
        with open(path+f'_{ngram}.x', 'r') as f:
            for i, line in enumerate(f):
                setattr(examples[i], f'{ngram}', ngram_field.preprocess(line.strip()))
            
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

#V = 10000
#import pdb; pdb.set_trace()
num_clusters = 256
# GPU check, make sure to use GPU where available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

## Turn off annoying torchtext warnings about pending deprecations
warnings.filterwarnings("ignore", module="torchtext", category=UserWarning)



ngram_fields = []
for ngram in range(NGRAMS_MIN, NGRAMS_MAX+1):
    ngram_field = tt.data.Field(include_lengths=True,         # include lengths
                        batch_first=False,            # batches will be max_len x batch_size
                        tokenize=lambda x: [int(item) for item in x.split()], # use split to tokenize
                        use_vocab=False,
                        pad_token=0,
                       ) 
    ngram_fields.append((f'ngram{ngram}', ngram_field))

#import pdb; pdb.set_trace()
text_field = tt.data.Field(include_lengths=True,
                    batch_first=False,            # batches will be max_len x batch_size
                    tokenize=lambda x: [item for item in x.split()], # use split to tokenize
                    )            # no <eos>

# Make splits for data
train_data, val_data, test_data= CatDataset.splits(
    text_field, ngram_fields, path=data_folder,
    train='train', validation='val', test='test')
#import pdb; pdb.set_trace()

# Build vocabulary
#SRC.build_vocab(train_data.z)
text_field.build_vocab(train_data.text)

#print (f"Size of src vocab: {len(SRC.vocab)}")
print (f"Size of tgt vocab: {len(text_field.vocab)}")
#print (f"Index for src padding: {SRC.vocab.stoi[SRC.pad_token]}")
pad_idx = text_field.vocab.stoi[text_field.pad_token]
print (f"Index for tgt padding: {text_field.vocab.stoi[text_field.pad_token]}")
#print (f"Index for start of sequence token: {TGT.vocab.stoi[TGT.init_token]}")
#print (f"Index for end of sequence token: {TGT.vocab.stoi[TGT.eos_token]}")


print ('batch size', BATCH_SIZE)
train_iter, val_iter, test_iter= tt.data.BucketIterator.splits((train_data, val_data, test_data),
                                                     batch_size=BATCH_SIZE, 
                                                     device=device,
                                                     repeat=False, 
                                                     sort_key=lambda x: len(x.text), # sort by length to minimize padding
                                                     sort_within_batch=True,
                                                     )


#import pdb; pdb.set_trace()
batch = next(iter(train_iter))
src, src_lengths = batch.ngram4
tgt, tgt_lengths = batch.text
print (f"Size of src batch: {src.shape}")
print (f"Third src sentence in batch: {src[:, 0]}")
print (f"Length of the third src sentence in batch: {src_lengths[0]}")
#print (f"Converted back to string: {' '.join([SRC.vocab.itos[i] for i in src[:, 2]])}")
print (f"Converted back to string: {' '.join([text_field.vocab.itos[i] for i in tgt[:, 0]])}")

print (src)
print (tgt)

#vocab_size_x = V + 1
#import pdb; pdb.set_trace()
vocab_size_x = len(text_field.vocab)
vocab_size_z = num_clusters + 1
vocab_size_ngrams = {}
for ngram in range(NGRAMS_MIN, NGRAMS_MAX+1):
    filename = os.path.join(data_folder, f'id2letters_{ngram}.txt')
    vocab_size_ngrams[ngram] = len(open(filename).readlines()) + 2
