import collections
import spacy
import tqdm
import json
import neuralcoref
import argparse


parser = argparse.ArgumentParser(description='Infer coreference chains using neuralcoref.')
parser.add_argument('--input_filename', type=str, required=True,
                    help='A json file containing articles about films.')
parser.add_argument('--output_filename', type=str, required=True,
                    help='Outputs a json file containing coreference chains.')
parser.add_argument('--start_of_sentence', type=str, default='.',
                    help='Default token for start-of-sentence in the coreference chain.')
args = parser.parse_args()


def is_person_pronoun(token):
    filtered_types = ['it', 'its']

    # 'cleaves' is sometimes incorrectly recognized by neuralcoref as a person
    filtered_types.append('cleaves') 

    if token.text.lower() in filtered_types:
        return False

    pronoun_types = set(['Ours', 'thy', 'We', 'Ourselves', 'your', 'Him', \
            'They', 'herself', 'you', 'myself', 'Myself', 'himself', 'Hers', \
            'ourselves', 'my', 'Me', 'themselves', 'Her', 'Our', 'yourself', \
            'her', 'ours', 'My', 'Your', 'He', 'Their', 'theirs', 'she', \
            'thee', 'he', 'Yours', 'Mine', 'mine', 'Theirs', 'Them', 'hers', \
            'him', 'I', 'us', 'our', 'Themselves', 'their', 'His', \
            'yourselves', 'his', 'they', 'Us', 'we', 'Himself', 'me', 'You', \
            'them', 'She'])
    if token.text in pronoun_types:
        return True

    if token.pos_ == 'PRON':
        return True

    return False


def is_person(token):
    return is_person_pronoun(token) or (token.ent_type_ == 'PERSON')


def get_mention_type(token):
    pronoun_types = {
            'MALE': set(['himself', 'him', 'he', 'his']),
            'FEMALE': set(['her', 'she', 'hers', 'herself']),
            'PLURAL': set(['them', 'our', 'they', 'yourselves', 'their', \
                    'us', 'you', 'theirs', 'we', 'ourselves', 'themselves', 'ours'])
    }

    for mention_type in pronoun_types:
        if token.text.lower() in pronoun_types[mention_type]:
            return mention_type

    return None


class Entity:
    def __init__(self, entity_id):
        self.entity_id = entity_id
        self.entity_type_counts = collections.defaultdict(int)
        self.is_person = False

    def add_mention(self, mention):
        self.is_person = self.is_person or is_person(mention)

        mention_type = get_mention_type(mention)

        if mention_type is not None:
            self.entity_type_counts[mention_type] += 1

    def get_entity_type(self):
        most_freq_type = None
        max_count = 0
        for entity_type in self.entity_type_counts:
            if self.entity_type_counts[entity_type] > max_count:
                max_count = self.entity_type_counts[entity_type]
                most_freq_type = entity_type
        return most_freq_type


def is_root_mention(token):
    entity_id = None
    flag = False
    if token._.in_coref:
        clusters = token._.coref_clusters
        for cluster in clusters:
            mentions = cluster.mentions
            for mention in mentions:
                root = mention.root
                if token.i == root.i:
                    entity_id = cluster.i
                    flag = True
    return (flag, entity_id)


def main(args):
    nlp = spacy.load('en_core_web_lg')
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    neuralcoref.add_to_pipe(nlp)

    samples = json.load(open(args.input_filename))
    start_of_sentence = args.start_of_sentence
    for sample in tqdm.tqdm(samples):
        sections = sample['sections']
        text = '\n\n'.join(sections).strip()
        doc = nlp(text)
    
        entity_dict = {}
        coreference_chain_tmp = []
        coreference_chain = []
        for token in doc:
            if token.is_sent_start:
                coreference_chain_tmp.append(start_of_sentence)

            # First, extract person entity mentions
            flag_root_mention, entity_id = is_root_mention(token)

            # Next, add mention to entity (to determine entity type for non-pronouns)
            if flag_root_mention:
                assert entity_id is not None
                if entity_id not in entity_dict:
                    entity_dict[entity_id] = Entity(entity_id)
                entity_dict[entity_id].add_mention(token)

                coreference_chain_tmp.append((token, entity_id))

        # Next, remove non-person entities
        coreference_chain_tmp = [item for item in coreference_chain_tmp \
                if item == start_of_sentence or entity_dict[item[1]].is_person]

        # Finally, replace non-pronouns with entity types
        for item in coreference_chain_tmp:
            if item == start_of_sentence:
                coreference_chain.append(item)
            else:
                token, entity_id = item
                entity = entity_dict[entity_id]

                if is_person_pronoun(token):
                    text = f'{token}'
                else:
                    entity_type = entity.get_entity_type()
                    text = entity_type
                coreference_chain.append(f'{text}:{entity_id}')

        sample['coreference_chain'] = coreference_chain

    json.dump(samples, open(args.output_filename, 'w'))
    

if __name__ == '__main__':
    main(args)
