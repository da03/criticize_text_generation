import torch
import sys
import argparse
import time

from data import load_data
from model import HSMMModel

import torch.nn as nn

parser = argparse.ArgumentParser(description='Train an HSMM language model.')
# Data
parser.add_argument('--dataset_folder', type=str, required=True,
                    help='Data folder.')
parser.add_argument('--vocab_folder', type=str, default='language_model_checkpoints/hsmm',
                    help='Vocabulary folder storing the constructed vocabulary files subseq_vocab_n-gram.txt.')
parser.add_argument('--pad_token', type=str, default='<pad>',
                    help='Special token reserved for padding.')
parser.add_argument('--unk_token', type=str, default='<unk>',
                    help='Special token reserved for unknown subsequences.')
# Loading/Saving
parser.add_argument('--checkpoint_folder', type=str, default='language_model_checkpoints/hsmm',
                    help='Folder storing the trained models.')
parser.add_argument('--train_from', type=str, default=None,
                    help='Train from a checkpoint.')
# Model parameters
parser.add_argument('--Z', type=int, default=800,
                    help='The number of states.')

# Optimization parameters
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
parser.add_argument('--save_every', type=int, default=90,
                    help='Save model every this number of updates.')
parser.add_argument('--decay_not_improving', type=int, default=8,
                    help='Decay learning rate when the printed stats do not improve this many times.')
parser.add_argument('--decay_factor', type=int, default=4,
                    help='Divide learning rate by this value during lr decay.')
parser.add_argument('--min_lr', type=float, default=3e-4,
                    help='Min learning rate.')
parser.add_argument('--accumulate', type=int, default=1,
                    help='Accumulate gradients this many times before updating parameters.')

def compute_loss_and_backward(model, x, x_lengths, subseq_ids, num_words, accumulate, backward=True):
    batch_size, seq_len = x.size()
    if (not backward) or (seq_len <= 390) or (batch_size <= 4):
        logits = model(x, x_lengths, subseq_ids)
        loss = -logits.sum()
        if backward:
            loss.div(num_words*accumulate).backward()
        return loss.item()
    else: # to avoid OOM errors we split a batch into two halves
        assert backward
        split = batch_size // 2
        loss1 = compute_loss_and_backward(model, x[:split].contiguous(), x_lengths[:split].contiguous(), \
                {k:subseq_ids[k][:split].contiguous() for k in subseq_ids}, num_words, accumulate)
        loss2 = compute_loss_and_backward(model, x[split:].contiguous(), x_lengths[split:].contiguous(), \
                {k:subseq_ids[k][split:].contiguous() for k in subseq_ids}, num_words, accumulate)
        return loss1 + loss2
            
def get_subseq_ids(batch, subseq_vocab_sizes):
    subseq_ids = {}
    for n in subseq_vocab_sizes:
        subseq_id, subseq_length = getattr(batch, f'{n}_gram')
        subseq_ids[n] = subseq_id
    return subseq_ids


def main(args):
    dataset_folder = args.dataset_folder
    vocab_folder = args.vocab_folder
    Z = args.Z
    batch_size = args.batch_size
    pad_token = args.pad_token
    unk_token = args.unk_token

    # Load data
    train_iter, val_iter, test_iter, subseq_vocab_sizes, x_pad_id, subseq_pad_id, subseq_unk_id \
            = load_data(dataset_folder, vocab_folder, Z, batch_size, pad_token, unk_token)

    # Build model
    model = HSMMModel(Z, subseq_vocab_sizes, subseq_pad_id=subseq_pad_id, subseq_unk_id=subseq_unk_id)
    model.cuda()

    if args.fix_z_transitions:
        model.fix_z_transitions()
    if args.fix_n_emissions:
        model.fix_n_emissions()

    # Load checkpoint if specified
    if args.train_from is not None:
        print (f'Loading checkpoint from {args.train_from}')
        model.load_state_dict(torch.load(args.train_from))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    import pdb; pdb.set_trace()

    # Train
    not_improving_since = 0
    num_updates = 0
    best_train_PPL = float('inf')
    best_val_PPL = float('inf')
    running_words = 0
    running_loss = 0.
    start_time = time.time()
    for epoch in range(args.epochs): 
        model.train()
        for batch in train_iter:
            x, x_lengths = batch.text
            num_words = x.ne(x_pad_id).sum().item()
            running_words += num_words
         
            subseq_ids = get_subseq_ids(batch, subseq_vocab_sizes)

            loss = compute_loss_and_backward(model, x, x_lengths, subseq_ids, num_words, args.accumulate)
            running_loss += loss
            num_updates += 1

            if num_updates % args.accumulate == 0:
                optimizer.step()
                optim.zero_grad()
            if num_updates % args.log_every == 0:
                PPL = math.exp(running_loss / running_words)
                print (f'Epoch {epoch} - #updates: {num_updates}, PPL: {PPL}, time: {time.time()-start_time}')
                sys.stdout.flush()
                if PPL < best_train_PPL:
                    best_train_PPL = PPL
                    not_improving_since = 0
                else:
                    not_improving_since += 1
                    if not_improving > args.decay_not_improving:
                        not_improving_since = 0
                        for g in optim.param_groups:
                            g['lr'] = max(g['lr'] / args.decay_factor, args.min_lr)
                            print (f'Decay lr to {g["lr"]}')

                if step % args.save_every == 0:
                    save_path = os.path.join(args.checkpoint_folder, 'checkpoint_last.pt')
                    torch.save(model.state_dict(), save_path)
                    print (f'Model saved to {save_pth}')

                running_words = 0
                running_loss = 0.
        print ('Validating')
        total_loss_val = 0.
        total_words_val = 0
        model.eval()
        with torch.no_grad():
            for batch in val_iter:
                x, x_lengths = batch.text
                num_words = x.ne(pad_idx).sum().item()
         
                total_words_val += num_words
                subseq_ids = get_subseq_ids(batch, subseq_vocab_sizes)
                loss = compute_loss_and_backward(model, x, x_lengths, subseq_ids, num_words, args.accumulate, backward=False)
                total_loss_val += loss.item()
        val_PPL = math.exp(total_loss_val / total_words_val)
        if val_PPL < best_val_PPL:
            save_path = os.path.join(args.checkpoint_folder, 'checkpoint_best.pt')
            torch.save(model.state_dict(), save_path)
            print (f'Model saved to {save_pth}')
            best_val_PPL = val_PPL
        print (f'End of Epoch {epoch}, val PPL: {val_PPL}, best val PPL: {best_val_PPL}')
        sys.stdout.flush()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
