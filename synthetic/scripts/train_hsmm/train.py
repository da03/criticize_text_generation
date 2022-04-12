import torch
import sys
import argparse

from data import load_data
from model import *

import torch.nn as nn

parser = argparse.ArgumentParser(description='Train an HSMM language model.')
parser.add_argument('--dataset_folder', type=str, required=True,
                    help='Data folder.')
parser.add_argument('--vocab_folder', type=str, default='language_model_checkpoints/hsmm',
                    help='Vocabulary folder storing the constructed vocabulary files subseq_vocab_n-gram.txt.')
parser.add_argument('--checkpoint_folder', type=str, default='language_model_checkpoints/hsmm',
                    help='Folder storing the trained models.')
parser.add_argument('--train_from', type=str, default=None,
                    help='Train from a checkpoint.')
parser.add_argument('--Z', type=int, default=800,
                    help='The number of states.')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=10,
                    help='The number of states.')
parser.add_argument('--lr', type=float, default=3e-1,
                    help='Learning rate.')
parser.add_argument('--fix_z_transitions', action='store_true',
                    help='Do not train the transition matrix.')
parser.add_argument('--fix_n_emissions', action='store_true',
                    help='Do not train the length emission matrix.')
parser.add_argument('--log_every', type=int, default=30,
                    help='Print training stats every this number of updates.')
parser.add_argument('--decay_not_improving', type=int, default=8,
                    help='Decay learning rate when the printed stats do not improve this many times.')
parser.add_argument('--accumulate', type=int, default=1,
                    help='Accumulate gradients this many times before updating parameters.')
parser.add_argument('--pad_token', type=str, default='<pad>',
                    help='Special token reserved for padding.')
parser.add_argument('--unk_token', type=str, default='<unk>',
                    help='Special token reserved for unknown subsequences.')

def main(args):
    dataset_folder = args.dataset_folder
    vocab_folder = args.vocab_folder
    Z = args.Z + 1 # plus one for initial state
    batch_size = args.batch_size
    pad_token = args.pad_token
    unk_token = args.unk_token

    # Load data
    train_iter, val_iter, test_iter, subseq_vocab_sizes, x_vocab_size \
            = load_data(dataset_folder, vocab_folder, Z, batch_size, pad_token, unk_token)

    # Build model
    model = HSMMModel(Z, x_vocab_size, subseq_vocab_sizes)
    model.cuda()

    if args.fix_z_transitions:
        model.fix_z_transitions()
    if args.fix_n_emissions:
        model.fix_n_emissions()

    # Load checkpoint if specified
    if args.train_from is not None:
        print (f'Loading checkpoint from {args.train_from}')
        model.load_state_dict(torch.load(args.train_from))

    not_improving = 0
    best_ppl_sofar = float('inf')
    vocab_size_z = args.vocab_size_z
    print ('vocab_size_z', vocab_size_z)
    sys.stdout.flush()
    loading = False
    if args.train_from != '':
        loading = True
        print ('loading', args.train_from)
        model.load_state_dict(torch.load(args.train_from), strict=False)
    #optim = torch.optim.SparseAdam(model.parameters(), lr=args.lr)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optim = get_std_opt(args.hidden_size, model.parameters())

    epochs = args.epochs
    best_val_ppl = float('inf')
    num_updates = 0
    #model.touched_ngram_ids = {}
    for epoch in range(epochs): 
      model.train()
      step = 0
      total_words = 0
      total_loss = 0.0
      #import pdb; pdb.set_trace()
      import time
      start_time = time.time()
      for batch in train_iter:
        step += 1
        num_updates += 1
        #z, z_lengths = batch.z
        x, x_lengths = batch.text
        #if x.size(0) > 350:
        #    continue
        x = x.to(device)
        x_in = x.transpose(0, 1).contiguous()
        x_out = x.transpose(0, 1).contiguous()
        num_words = x_out.ne(pad_idx).sum().item()
        total_words += num_words
       
        batch_size = x.size(-1)
        seq_len = x.size(0)

        ngram_ids = {}
        for ngram in range(NGRAMS_MIN, NGRAMS_MAX+1):
            ngram_id, ngram_length = getattr(batch, f'ngram{ngram}')
            ngram_ids[ngram] = ngram_id.transpose(0, 1).contiguous().cpu()
        #import pdb; pdb.set_trace()
        print (f'{x_out.shape}')
        sys.stdout.flush()

        if x_out.size(1) > 390 and x_out.size(0) > 4:
          split_size = x_out.size(0) // 2
          print ('splitting')

          logits = model(x_out[:split_size].contiguous(), x_lengths[:split_size].contiguous(), {k:ngram_ids[k][:split_size].contiguous() for k in ngram_ids})
          loss = -logits.sum()
          total_loss += loss.item()
          loss.div(num_words*args.accumulate).backward()

          logits = model(x_out[split_size:].contiguous(), x_lengths[split_size:].contiguous(), {k:ngram_ids[k][split_size:].contiguous() for k in ngram_ids}, dropout=args.dropout)
          loss = -logits.sum()
          total_loss += loss.item()
          loss.div(num_words*args.accumulate).backward()
        else:
          logits = model(x_out, x_lengths, ngram_ids, dropout=args.dropout)
          loss = -logits.sum()
          total_loss += loss.item()
          #import pdb; pdb.set_trace()
          loss.div(num_words*args.accumulate).backward()
        if num_updates % args.accumulate == 0:
          #import pdb; pdb.set_trace()
          #logsumexps_old = model.get_logsumexps()
          optim.step()
          #logsumexps_new = model.get_logsumexps()
          #for l in model.touched_ngram_ids:
          #    log_partition = model.log_partitions[l]
          #    new_logp = logsumexps_new[l]
          #    old_logp = logsumexps_old[l]
          #    mask = new_logp.gt(old_logp) # vocab_size_z
          #    log_a_minus_b = new_logp + (1 - (old_logp - new_logp).exp()).log()
          #    log_partition_new = torch.stack((log_partition, log_a_minus_b), 0).logsumexp(0)
          #    log_partition_new[~mask] = 0
          #    log_b_minus_a = old_logp + (1 - (new_logp - old_logp).exp()).log()
          #    log_partition_new2 = log_partition + ( 1 - (log_b_minus_a - log_partition).exp()).log()
          #    log_partition_new2[mask] = 0
          #    model.log_partitions[l] = log_partition_new + log_partition_new2
          #model.touched_ngram_ids = {}
          #if num_updates % (30*args.accumulate) == 0:
          #    model.reset_partition()

              #log_partition_new = torch.stack((log_partition, logsumexps_new[l]), 0).logsumexp(0)
              #model.log_partitions[l] = log_partition_new + (1 - (logsumexps_old[l] - log_partition_new[l]).exp()).log()
              #partition = model.log_partitions[l].exp()
              #partition += logsumexps_new[l].exp() - logsumexps_old[l].exp()
              #model.log_partitions[l] = partition.log()
          optim.zero_grad()
        #logsumexps_new = model.get_logsumexps()
        print (f' - Epoch: {epoch}, step: {step}, ppl: {math.exp(min(10, total_loss / total_words))}, time: {time.time()-start_time}')
        sys.stdout.flush()
        if step % args.print_every == 0:
            print (f'Epoch: {epoch}, step: {step}, ppl: {math.exp(total_loss / total_words)}, time: {time.time()-start_time}')
            ppl = math.exp(total_loss / total_words)
            if ppl < best_ppl_sofar:
                best_ppl_sofar = ppl
                not_improving = 0
            else:
                not_improving += 1
                if not_improving > 8:
                #if not_improving > 32:
                    print ('NOT IMPROVING')
                    not_improving = 0
                    if decay_lr:
                        for g in optim.param_groups:
                            g['lr'] = max(g['lr'] / 4.0, 3e-4)
                            #g['lr'] = max(g['lr'] / 4.0, 1e-3)
                            print ('NEW lr', g['lr'])

            sys.stdout.flush()
            #x_in = x.new(x.size(0), 1).fill_(0)
            #import pdb; pdb.set_trace()
            #s = sample(model.gpt, x_in, 1000, num_zs=50, z_delimiter=TGT.vocab.stoi['<space>'])
            #for x in s:
            #    print (f"Sample: {' '.join([TGT.vocab.itos[i] for i in x])}")
            if step % (3*args.print_every) == 0:
                print ('saving')
                #save_path = f'/n/holyscratch01/rush_lab/Users/yuntian/working_effn_lr{args.lr}_b{BATCH_SIZE}_acc{args.accumulate}_C{vocab_size_z}_xaiver_dropout{args.dropout}_trainz{train_z}_load{loading}_decay{decay_lr}_trainl{train_l}_init{init}_imprv8_3e4_noon_8_3e4.pt'
                save_path = f'/n/holyscratch01/rush_lab/Users/yuntian/mar3/unigram_working_effn_lr{args.lr}_b{BATCH_SIZE}_acc{args.accumulate}_C{vocab_size_z}_dropout{args.dropout}_trainz{train_z}_load{loading}_decay{decay_lr}_trainl{train_l}_init{init}_imprv8_3e4_noon_8_3e4.pt'
                print (save_path)
                torch.save(model.state_dict(), save_path)
                #model.reset_partition()

            sys.stdout.flush()
            total_words = 0
            total_loss = 0.
      #  if num_updates > 90:
      #      sys.exit(0)
      print ('Validation')
      total_loss_val = 0.
      total_words_val = 0
      model.eval()
      with torch.no_grad():
        for batch in val_iter:
          x, x_lengths = batch.text
          x = x.to(device)
          x_in = x.transpose(0, 1).contiguous()
          x_out = x.transpose(0, 1).contiguous()
          num_words = x_out.ne(pad_idx).sum().item()
       
          batch_size = x.size(-1)
          seq_len = x.size(0)
          total_words_val += num_words
         
          ngram_ids = {}
          for ngram in range(NGRAMS_MIN, NGRAMS_MAX+1):
              ngram_id, ngram_length = getattr(batch, f'ngram{ngram}')
              ngram_ids[ngram] = ngram_id.transpose(0, 1).contiguous().cpu()
          #logits = model(x_out, x_lengths)
          logits = model(x_out, x_lengths, ngram_ids)
          loss = -logits.sum()#loss_fn(logits.view(-1, vocab_size_x), x_out.view(-1))
          total_loss_val += loss.item()
      if math.exp(total_loss_val / total_words_val) < best_val_ppl:
          print ('not saving')
          save_path = f'/n/holyscratch01/rush_lab/Users/yuntian/mar3/unigram_working_eff_best_{args.lr}_{BATCH_SIZE}_new_trunc_acc_{args.accumulate}_{vocab_size_z}_fixgrad_xaiver_dropout{args.dropout}_{train_z}_wed_{loading}_thu_{decay_lr}_{train_l}_fri_{init}.pt'
          print (save_path)
          #torch.save(model.state_dict(), save_path)
      best_val_ppl = min(best_val_ppl, math.exp(total_loss_val / total_words_val))
      print (f'End of Epoch {epoch}, val ppl: {math.exp(total_loss_val / total_words_val)}, best val ppl: {best_val_ppl}')
      sys.stdout.flush()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
