import collections

def relabel_ngram(ngram):
    """Relabel entity ids in the ngram to avoid data sparsity issues. For example,
    if the original ngram is "Female:42 . She:42 Male:34", then the relabeled ngram
    will be "Female:0 . She:0 Male:1".
    """
    entity_ids = {}
    ngram_relabeled = []
    for w in ngram:
        items = w.split(':')
        if len(items) == 1:
            ngram_relabeled.append(w)
        else:
            assert len(items) == 2
            text, entity_id = items
            if entity_id not in entity_ids:
                entity_ids[entity_id] = len(entity_ids)
            ngram_relabeled.append(f'{text}:{entity_ids[entity_id]}')
    return tuple(ngram_relabeled)

class NGramCritic():
    def __init__(self, N):
        self.N = N

    def get_prob_kenlm_recursive(self, prefix, suffix, n):
        ngram_prefix_count = getattr(self, f'ngram_prefix_count_{n}')
        #prefix, suffix = self.get_condition_suffix(ngram)
        ngram_prefix_and_suffix_count = getattr(self, f'ngram_prefix_and_suffix_count_{n}')
        ngram_possible_suffix_given_prefix = getattr(self, f'ngram_possible_suffix_given_prefix_{n}')
        delta = 0.75
        N1 = self.N1s[n]
        N2 = self.N2s[n]
        #import pdb; pdb.set_trace()
        delta = N1 / (N1+2*N2)
        #delta = 0.9
        if n == 1:
            delta = 0
        #assert prefix in ngram_prefix_count, (prefix, n)
        #assert ngram_prefix_count[prefix] > 0
        first_term = max(ngram_prefix_and_suffix_count[(prefix, suffix)] - delta, 0) / ngram_prefix_count[prefix]
        weight = delta*len(ngram_possible_suffix_given_prefix[prefix]) / ngram_prefix_count[prefix]
        if n > 1:
            second_term = self.get_prob_kenlm_recursive(prefix[1:], suffix, n-1)
        else:
            if first_term <= 0:
                first_term = 1e-20
            assert first_term > 0, (prefix, suffix)
            second_term = 0
        return first_term + weight * second_term
    def get_prob_kenlm(self, line):
        #self.vocab_size = len(self.vocab) * self.ngram
        items = line.strip().split()
        ngrams = generate_N_grams(items, self.ngram_max)
        total_log_prob = 0
        total = 0
        log_probs = []
        for ngram in ngrams:
            #import pdb; pdb.set_trace()
            prefix, suffix = self.get_condition_suffix(ngram)
            #if suffix == '<start-of-sentence>':
            #    continue
            order = self.ngram_max
            order = 5
            prefix = prefix[(self.ngram_max-order):]
            ngram_prefix_count = getattr(self, f'ngram_prefix_count_{order}')
            #import pdb; pdb.set_trace()
            while prefix not in ngram_prefix_count:
                order -= 1
                ngram_prefix_count = getattr(self, f'ngram_prefix_count_{order}')
                prefix = prefix[1:]
                assert order >= 0, prefix
                #ngram = ngram[1:]
                #prefix, suffix = self.get_condition_suffix(ngram)
            #print ('-**', prefix, order)
            prob = self.get_prob_kenlm_recursive(prefix, suffix, order)
            if True or order == self.ngram_max:
                if True and math.log(prob) < -7:
                    ngram_prefix_and_suffix_count = getattr(self, f'ngram_prefix_and_suffix_count_{order}')
                    ngram_possible_suffix_given_prefix = getattr(self, f'ngram_possible_suffix_given_prefix_{order}')
                    max_score = -float('inf')
                    max_w = None
                    #import pdb; pdb.set_trace()
                    for w in ngram_possible_suffix_given_prefix[prefix]:
                        if w == '<start-of-sentence>':
                            continue
                        score = self.get_prob_kenlm_recursive(prefix,w, order)
                        score = math.log(score)
                        if score > max_score:
                            max_score = score
                            max_w = w
                    #print (ngram, math.log(prob), max_w, max_score)
                    #print (prefix, suffix, math.log(prob), max_w, max_score, order)
                    print ('\t'.join([str(item) for item in list(prefix) + [suffix,] + [math.log(prob),] + [max_w,] + [max_score,] + [order,]]))
                    log_probs.append(f'{math.log(prob)},{suffix},{max_w},{max_score},{order}')
                    total_log_prob += math.log(prob)
                    total += 1
                    continue

            #denominator = self.ngram_prefix_count[prefix] + self.smoothing
            #numerator = self.ngram_prefix_and_suffix_count[(prefix, suffix)] + self.smoothing
            #prob = numerator / denominator
            if prob <=0:
                import pdb; pdb.set_trace()
            #if suffix == '<start-of-sentence>':
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

    def fit(self, filename):
        self.N1s = collections.defaultdict(int) # ngrams that appeared once, this is for calculating smoothing factor
        self.N2s = collections.defaultdict(int) # ngrams that appeared twice

        samples = json.load(open(filename))
        N = self.N
        for n in range(1, N+1):
            # ngram = prefix (n-1 words) + suffix (1 word)
            setattr(self, f'ngram_prefix_count_{n}', collections.defaultdict(int)) # count prefix
            setattr(self, f'ngram_possible_suffix_given_prefix_{n}', collections.defaultdict(set)) # possible suffix given prefix
            setattr(self, f'ngram_prefix_and_suffix_count_{n}', collections.defaultdict(int)) # count (prefix, suffix)

            ngram_prefix_count = getattr(self, f'ngram_prefix_count_{n}')
            ngram_possible_suffix_given_prefix = getattr(self, f'ngram_possible_suffix_given_prefix_{n}')
            ngram_prefix_and_suffix_count = getattr(self, f'ngram_prefix_and_suffix_count_{n}')

            for sample in samples:
                coreference_chain = sample['coreference_chain']
                ngrams = generate_N_grams(coreference_chain, N)
                for ngram in ngrams:
                    ngram_relabeled = relabel_ngram(ngram)
                    ngram_relabeled = ngram_relabeled[(N-n):] # for low-order n we need to truncate
                    prefix, suffix = ngram_relabeled[:-1], ngram_relabeled[-1:]
                    ngram_prefix_count[prefix] += 1
                    ngram_possible_suffix_given_prefix[prefix].add(suffix)
                    ngram_prefix_and_suffix_count[(prefix, suffix)] += 1
            for ngram in ngram_prefix_and_suffix_count:
                if ngram_prefix_and_suffix_count[ngram] == 1:
                    self.N1s[n] += 1
                if ngram_prefix_and_suffix_count[ngram] == 2:
                    self.N2s[n] += 1
