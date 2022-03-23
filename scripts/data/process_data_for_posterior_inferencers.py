import os
import json
import argparse
import random

parser = argparse.ArgumentParser(description='Process raw json files to generate training data for posterior inferencers.')
parser.add_argument('--dataset_folder', type=str, required=True,
                    help='Folder containing train.json, val.json, and test.json.')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed.')
args = parser.parse_args()


def process_data_w_title(filename, filename_out, shuffle=False):
    data = json.load(open(filename))
    texts_labels = []
    with open(filename_out, 'w') as fout:
        for example in data:
            sections = example['sections']
            section_names = example['section_names']
            assert all(['\n\n' not in section_text for section_text in sections])
            for section_name, section in zip(section_names, sections):
                texts_labels.append((section, section_name))
        if shuffle:
            random.shuffle(texts_labels)
        for section, section_name in texts_labels:
            fout.write(json.dumps({'label':section_name, 'text':section}) + '\n')

def main(args):
    dataset_folder = args.dataset_folder
    for split in ['train', 'val', 'test']:
        random.seed(args.seed)
        filename = os.path.join(dataset_folder, f'{split}.json')
        filename_out = os.path.join(dataset_folder, f'{split}.posterior_inferencer.json')
        process_data_w_title(filename, filename_out, shuffle=split=='train')


if __name__ == '__main__':
    main(args)
