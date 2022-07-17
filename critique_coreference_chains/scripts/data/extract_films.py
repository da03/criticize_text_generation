import spacy
import tqdm
import sys, os
import json
import re
import argparse

parser = argparse.ArgumentParser(description='Filter all articles that are not about films.')
parser.add_argument('--input_filename', type=str, required=True,
                    help='A json file containing articles (either a data split or generated by an LM).')
parser.add_argument('--output_filename', type=str, required=True,
                    help='Outputs a json file containing only articles about films.')
args = parser.parse_args()


def main(args):
    nlp = spacy.load("en_core_web_lg")
    input_filename = args.input_filename
    output_filename = args.output_filename
    samples = json.load(open(input_filename))
    samples_films = []
    for article in tqdm.tqdm(samples):
        sections = article['sections']
        first_section = sections[0]
        doc = nlp(first_section)
        if len(list(doc.sents)) == 0:
            continue
        first_sentence = list(doc.sents)[0]
        words = [f'{word}' for word in list(first_sentence)]
    
        if 'film' in words and ('born' not in words):
            text = f'{first_sentence}'
            m = re.match(r'.*is a.*film.*', text)
            if m:
                samples_films.append(article)
    json.dump(samples_films, open(output_filename, 'w'))
    

if __name__ == '__main__':
    main(args)
