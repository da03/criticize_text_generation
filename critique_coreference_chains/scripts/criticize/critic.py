import collections
import json
import pickle

def get_N_grams(words, ngram, bos='<bos>', eos='<eos>'):
    words = [bos] * (ngram-1) + words + [eos]
    ngrams = list(zip(*[words[i:] for i in range(ngram)]))
    return ngrams

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
    def __init__(self, N, unk_prob=1e-20):
        self.N = N
        self.unk_prob = 1e-20

    def get_word_prob(self, prefix, suffix, n):
        ngram_prefix_count = getattr(self, f'ngram_prefix_count_{n}')
        ngram_prefix_and_suffix_count = getattr(self, f'ngram_prefix_and_suffix_count_{n}')
        ngram_possible_suffix_given_prefix = getattr(self, f'ngram_possible_suffix_given_prefix_{n}')

        # compute smoothing factor as n1 / (n1 + 2*n2)
        # (https://www.cs.cmu.edu/afs/cs/project/cmt-55/lti/Courses/731/homework/srilm/man/html/ngram-discount.7.html)
        N1 = self.N1s[n]
        N2 = self.N2s[n]
        delta = N1 / (N1 + 2*N2)
        if n == 1:
            delta = 0

        # compute the actual probability using recursion
        first_term = max(ngram_prefix_and_suffix_count[(prefix, suffix)] - delta, 0) / ngram_prefix_count[prefix]
        weight = delta * len(ngram_possible_suffix_given_prefix[prefix]) / ngram_prefix_count[prefix]
        if n == 1:
            if first_term <= 0: # unobserved unigrams, should be rare
                first_term = self.unk_prob
            return first_term
        else:
            second_term = self.get_word_prob(prefix[1:], suffix, n-1)
            return first_term + weight*second_term

    def get_coreference_chain_logprob(self, coreference_chain):
        ngrams = get_N_grams(coreference_chain, self.N)

        total_logprob = 0
        total_normalizer = 0
        for ngram in ngrams:
            ngram_relabeled = relabel_ngram(ngram)
            prefix, suffix = ngram_relabeled[:-1], ngram_relabeled[-1]

            # backoff to an order when prefix can be found
            n = self.N
            ngram_prefix_count = getattr(self, f'ngram_prefix_count_{n}')
            while prefix not in ngram_prefix_count:
                n -= 1
                ngram_prefix_count = getattr(self, f'ngram_prefix_count_{n}')
                prefix = prefix[1:]
                assert n >= 0, prefix

            # compute prob of the given suffix
            prob = self.get_word_prob(prefix, suffix, n)

            total_logprob += math.log(prob)
            total_normalizer += 1

        return total_logprob, total_normalizer

    def evaluate_latent_PPL(self, filename):
        samples = json.load(open(filename))
        total_logprob, total_normalizer = 0, 0
        for sample in samples:
            coreference_chain = sample['coreference_chain']
            logprob, normalizer = self.get_coreference_chain_logprob(coreference_chain)
            total_logprob += logprob
            total_normalizer += normalizer
        return math.exp(-total_logprob / total_normalizer)

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
                ngrams = get_N_grams(coreference_chain, N)
                for ngram in ngrams:
                    ngram_relabeled = relabel_ngram(ngram)
                    ngram_relabeled = ngram_relabeled[(N-n):] # for low-order n we need to truncate
                    prefix, suffix = ngram_relabeled[:-1], ngram_relabeled[-1]
                    ngram_prefix_count[prefix] += 1
                    ngram_possible_suffix_given_prefix[prefix].add(suffix)
                    ngram_prefix_and_suffix_count[(prefix, suffix)] += 1
            for ngram in ngram_prefix_and_suffix_count:
                if ngram_prefix_and_suffix_count[ngram] == 1:
                    self.N1s[n] += 1
                if ngram_prefix_and_suffix_count[ngram] == 2:
                    self.N2s[n] += 1

    def save(self, output_critic_filename):
        pickle.dump(self, open(output_critic_filename, 'wb'))
