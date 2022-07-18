import sys
import os
import math
import json
import argparse

import torch

parser = argparse.ArgumentParser(description='Fit a critic P_c(z) on the training data.')
parser.add_argument('--train_coreference_chains', type=str, required=True,
                    help='A json file containing coreference chains.')
parser.add_argument('--output_critic_filename', type=str, required=True,
                    help='Output file for the learned critic.')
args = parser.parse_args()


def generate_N_grams(words, ngram, bos='<bos>', eos='<eos>'):
    words = [bos] * (ngram-1) + words + [eos]
    ngrams = list(zip(*[words[i:] for i in range(ngram)]))
    return ngrams

class NGramLM():
    def __init__(self, ngram=2, smoothing=1):
        self.ngram_c_dict = collections.defaultdict(int)
        self.ngram_c_word_dict = collections.defaultdict(int)
        self.vocab = set([])
        self.ngram = ngram
        self.smoothing = smoothing

    def get_condition_next_word(self, ngram):
        ngram_c = ngram[:-1]
        next_word = ngram[-1]
        ngram_reverse = []
    
        items = ngram[0].split(':')
    
        entity_ids = {}
        if len(items) == 1:
            current_entity_id = None
        else:
            assert len(items) == 2
            current_entity_id = items[1]
            entity_ids[current_entity_id] = len(entity_ids)
        #ngram_reverse.append(f'{text}:{entity_ids[entity_id]}')
        for w in ngram_c:
            items = w.split(':')
            if len(items) == 1:
                ngram_reverse.append(w)
            else:
                assert len(items) == 2
                text, entity_id = items
                if entity_id not in entity_ids:
                    entity_ids[entity_id] = len(entity_ids)
                ngram_reverse.append(f'{text}:{entity_ids[entity_id]}')
        ngram_c = tuple(ngram_reverse)
        #ngram_c_dict[ngram_c] += 1
        #denominator = self.ngram_c_dict[ngram_c] + self.vocab_size * self.smoothing
        items = next_word.split(':')
        if len(items) == 1:
            next_word = next_word
        else:
            assert len(items) == 2
            text, entity_id = items
            if entity_id not in entity_ids:
                entity_ids[entity_id] = len(entity_ids)
            next_word = f'{text}:{entity_ids[entity_id]}'
        #if ngram_c in self.ngram_c_dict:
        #    if (ngram_c, next_word) not in self.ngram_c_word_dict:
        #        print (ngram_c, next_word)
        #numerator = self.ngram_c_word_dict[(ngram_c, next_word)] + self.smoothing
        return ngram_c, next_word

    def get_prob(self, line):
        self.vocab_size = len(self.vocab) * self.ngram
        items = line.strip().split()
        ngrams = generate_N_grams(items, self.ngram)
        total_log_prob = 0
        total = 0
        log_probs = []
        for ngram in ngrams:
            ngram_c, next_word = self.get_condition_next_word(ngram)
            denominator = self.ngram_c_dict[ngram_c] + self.smoothing
            numerator = self.ngram_c_word_dict[(ngram_c, next_word)] + self.smoothing
            prob = numerator / denominator
            total_log_prob += math.log(prob)
            log_probs.append(math.log(prob))
            total += 1
        return total_log_prob, total, log_probs
    def get_prob_kenlm_recursive(self, ngram_c, next_word, n):
        ngram_c_dict = getattr(self, f'ngram_c_dict_{n}')
        #ngram_c, next_word = self.get_condition_next_word(ngram)
        ngram_c_word_dict = getattr(self, f'ngram_c_word_dict_{n}')
        ngram_c_w_dict = getattr(self, f'ngram_c_w_dict_{n}')
        delta = 0.75
        N1 = self.N1s[n]
        N2 = self.N2s[n]
        #import pdb; pdb.set_trace()
        delta = N1 / (N1+2*N2)
        #delta = 0.9
        if n == 1:
            delta = 0
        #assert ngram_c in ngram_c_dict, (ngram_c, n)
        #assert ngram_c_dict[ngram_c] > 0
        first_term = max(ngram_c_word_dict[(ngram_c, next_word)] - delta, 0) / ngram_c_dict[ngram_c]
        weight = delta*len(ngram_c_w_dict[ngram_c]) / ngram_c_dict[ngram_c]
        if n > 1:
            second_term = self.get_prob_kenlm_recursive(ngram_c[1:], next_word, n-1)
        else:
            if first_term <= 0:
                first_term = 1e-20
            assert first_term > 0, (ngram_c, next_word)
            second_term = 0
        return first_term + weight * second_term
    #def get_prob_kenlm_recursive2(self, ngram, n):
    #    ngram_c_dict = getattr(self, f'ngram_c_dict_{n}')
    #    ngram_c, next_word = self.get_condition_next_word(ngram)
    #    ngram_c_word_dict = getattr(self, f'ngram_c_word_dict_{n}')
    #    ngram_c_w_dict = getattr(self, f'ngram_c_w_dict_{n}')
    #    delta = 0.0
    #    first_term = max(ngram_c_word_dict[(ngram_c, next_word)] - delta, 0) / ngram_c_dict[ngram_c]
    #    weight = delta*len(ngram_c_w_dict[ngram_c]) / ngram_c_dict[ngram_c]
    #    if n > 1:
    #        second_term = self.get_prob_kenlm_recursive(ngram[1:], n-1)
    #    else:
    #        if first_term <= 0:
    #            first_term = 1e-20
    #        assert first_term > 0, (ngram_c, next_word)
    #        second_term = 0
    #    return first_term + weight * second_term
    def get_prob_kenlm(self, line):
        #self.vocab_size = len(self.vocab) * self.ngram
        items = line.strip().split()
        ngrams = generate_N_grams(items, self.ngram_max)
        total_log_prob = 0
        total = 0
        log_probs = []
        for ngram in ngrams:
            #import pdb; pdb.set_trace()
            ngram_c, next_word = self.get_condition_next_word(ngram)
            #if next_word == '<start-of-sentence>':
            #    continue
            order = self.ngram_max
            order = 5
            ngram_c = ngram_c[(self.ngram_max-order):]
            ngram_c_dict = getattr(self, f'ngram_c_dict_{order}')
            #import pdb; pdb.set_trace()
            while ngram_c not in ngram_c_dict:
                order -= 1
                ngram_c_dict = getattr(self, f'ngram_c_dict_{order}')
                ngram_c = ngram_c[1:]
                assert order >= 0, ngram_c
                #ngram = ngram[1:]
                #ngram_c, next_word = self.get_condition_next_word(ngram)
            #print ('-**', ngram_c, order)
            prob = self.get_prob_kenlm_recursive(ngram_c, next_word, order)
            if True or order == self.ngram_max:
                if True and math.log(prob) < -7:
                    ngram_c_word_dict = getattr(self, f'ngram_c_word_dict_{order}')
                    ngram_c_w_dict = getattr(self, f'ngram_c_w_dict_{order}')
                    max_score = -float('inf')
                    max_w = None
                    #import pdb; pdb.set_trace()
                    for w in ngram_c_w_dict[ngram_c]:
                        if w == '<start-of-sentence>':
                            continue
                        score = self.get_prob_kenlm_recursive(ngram_c,w, order)
                        score = math.log(score)
                        if score > max_score:
                            max_score = score
                            max_w = w
                    #print (ngram, math.log(prob), max_w, max_score)
                    #print (ngram_c, next_word, math.log(prob), max_w, max_score, order)
                    print ('\t'.join([str(item) for item in list(ngram_c) + [next_word,] + [math.log(prob),] + [max_w,] + [max_score,] + [order,]]))
                    log_probs.append(f'{math.log(prob)},{next_word},{max_w},{max_score},{order}')
                    total_log_prob += math.log(prob)
                    total += 1
                    continue

            #denominator = self.ngram_c_dict[ngram_c] + self.smoothing
            #numerator = self.ngram_c_word_dict[(ngram_c, next_word)] + self.smoothing
            #prob = numerator / denominator
            if prob <=0:
                import pdb; pdb.set_trace()
            #if next_word == '<start-of-sentence>':
            #    continue
            total_log_prob += math.log(prob)
            log_probs.append(math.log(prob))
            total += 1
        return total_log_prob, total, log_probs
    def read_val(self, filename):
        with open(filename) as fin:
            with open(filename + '.logprobs', 'w') as fout:
                ppls = []
                total_log_prob, total_num = 0, 0
                for line in fin:
                    log_prob, num, log_probs = self.get_prob_kenlm(line)
                    fout.write(' '.join([str(item) for item in log_probs]) + '\n')
                    #ppls.append(log_prob / num)
                    #ppls.extend([log_prob / num] * num)
                    ppls.extend([float(str(item).split(',')[0]) for item in log_probs])
                    total_log_prob += log_prob
                    total_num += num
                return math.exp(-total_log_prob / total_num), np.array(ppls)

    def read_train(self, filename):
        with open(filename) as fin:
            for line in fin:
                #import pdb; pdb.set_trace()
                items = line.strip().split()
                ngrams = generate_N_grams(items, self.ngram)
                for ngram in ngrams:
                    ngram_c, next_word = self.get_condition_next_word(ngram)
                    for w in ngram:
                        items = w.split(':')
                        if len(items) > 1:
                            self.vocab.add(items[0])
                        else:
                            self.vocab.add(w)
                    self.ngram_c_dict[ngram_c] += 1
                    self.ngram_c_word_dict[(ngram_c, next_word)] += 1
    def read_train_kenlm(self, filename, ngram_min=1, ngram_max=5):
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.N1s = collections.defaultdict(int)
        self.N2s = collections.defaultdict(int)
        lines = open(filename).readlines()
        for n in range(ngram_min, ngram_max+1):
            setattr(self, f'ngram_c_dict_{n}', collections.defaultdict(int))
            setattr(self, f'ngram_c_w_dict_{n}', collections.defaultdict(set))
            setattr(self, f'ngram_c_word_dict_{n}', collections.defaultdict(int))
            setattr(self, f'ngram_dict_{n}', collections.defaultdict(int))
            ngram_c_dict = getattr(self, f'ngram_c_dict_{n}')
            ngram_c_word_dict = getattr(self, f'ngram_c_word_dict_{n}')
            ngram_c_w_dict = getattr(self, f'ngram_c_w_dict_{n}')
            ngram_dict = getattr(self, f'ngram_dict_{n}')
            #with open(filename) as fin:
            for line in lines:
                #import pdb; pdb.set_trace()
                items = line.strip().split()
                #ngrams = generate_N_grams(items, n)
                #if n == 4:
                #    import pdb; pdb.set_trace()
                ngrams = generate_N_grams(items, ngram_max)
                for ngram in ngrams:
                    ngram_c, next_word = self.get_condition_next_word(ngram)
                    ngram_c = ngram_c[(ngram_max - n):]
                    ngram_dict[tuple(list(ngram_c) + [next_word])] += 1
                    #if next_word == '<start-of-sentence>':
                    #    continue
                    for w in ngram:
                        items = w.split(':')
                        if len(items) > 1:
                            self.vocab.add(items[0])
                        else:
                            self.vocab.add(w)
                    ngram_c_dict[ngram_c] += 1
                    ngram_c_word_dict[(ngram_c, next_word)] += 1
                    ngram_c_w_dict[ngram_c].add(next_word)
            for ngram in ngram_dict:
                if ngram_dict[ngram] == 1:
                    self.N1s[n] += 1
                if ngram_dict[ngram] == 2:
                    self.N2s[n] += 1

def main(args):
    critic = NGramLM(ngram=4, smoothing=0.01)
    critic.read_train_kenlm(train_file)
    critic.save(args.output_critic_filename)

if __name__ == '__main__':
    main(args)
