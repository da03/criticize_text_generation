import sys
import os
import json
import argparse
import random

parser = argparse.ArgumentParser(description='Process raw json files to generate training data for language models.')
parser.add_argument('--dataset_folder', type=str, required=True,
                    help='Folder containing train.json, val.json, and test.json.')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed.')
args = parser.parse_args()


def process_data_w_title(filename, filename_out_w_title, shuffle=False):
    data = json.load(open(filename))
    texts = []
    with open(filename_out_w_title, 'w') as fout:
        for example in data:
            sections = example['sections']
            section_names = example['section_names']
            assert all(['\n\n' not in section_text for section_text in sections])
            text = '\n\n'.join([f'{section_name}: {section_text}' for section_name, section_text in zip(section_names, sections)])
            texts.append(text)
        if shuffle:
            random.shuffle(texts)
        for text in texts:
            fout.write(f'<|endoftext|>{text}<|endoftext|>\n')


def process_data_wo_title(filename, filename_out_w_title, shuffle=False):
    data = json.load(open(filename))
    texts = []
    with open(filename_out_w_title, 'w') as fout:
        for example in data:
            sections = example['sections']
            section_names = example['section_names']
            assert all(['\n\n' not in section_text for section_text in sections])
            text = '\n\n'.join([f'{section_text}' for section_text in sections])
            texts.append(text)
        if shuffle:
            random.shuffle(texts)
        for text in texts:
            fout.write(f'<|endoftext|>{text}<|endoftext|>\n')


def main(args):
    dataset_folder = args.dataset_folder
    for split in ['train', 'val', 'test']:
        random.seed(args.seed)
        filename = os.path.join(dataset_folder, f'{split}.json')
        filename_out_w_title = os.path.join(dataset_folder, f'{split}.w_title.txt')
        filename_out_wo_title = os.path.join(dataset_folder, f'{split}.wo_title.txt')
        process_data_w_title(filename, filename_out_w_title, shuffle=split=='train')
        process_data_wo_title(filename, filename_out_wo_title, shuffle=split=='train')


if __name__ == '__main__':
    main(args)
