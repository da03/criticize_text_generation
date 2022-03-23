import sys
import json
import math
import argparse

from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser(description='Infer section titles based on section texts.')
parser.add_argument('--posterior_inferencer_checkpoint', type=str, required=True,
                    help='Posterior inferencer checkpoint. Should be a folder containing pytorch_model.bin and other configuration files.')
parser.add_argument('--input_file', type=str, required=True,
                    help='Input filename containing LM samples/real data.')
parser.add_argument('--output_file', type=str, required=True,
                    help='Output filename. The output file will contain the predicted section titles.')
args = parser.parse_args()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = AutoModelForSequenceClassification.from_pretrained('/n/holyscratch01/rush_lab/Users/yuntian/hierarchy/pubmed-final-2k-dataset/bert2e5real')
    model.cuda()
    model.eval()

    label2id = model.config.label2id
    id2label = model.config.id2label
   
    data = json.load(open(args.input_file))
    total = 0
    same = 0
    for sample in tqdm(data):
        sections = sample['sections']
        section_names = sample['section_names']
        predicted_section_names = []
        for section, section_name in zip(sections, section_names):
            inputs = tokenizer(section, padding='max_length', max_length=512, truncation=True)
            for k in inputs:
                inputs[k] = torch.LongTensor(inputs[k]).cuda().unsqueeze(0)
            outputs = model(**inputs)
            logits = outputs['logits'][0]
            label = logits.argmax(0).item()
            predicted_section_name = id2label[label]
            predicted_section_names.append(predicted_section_name)
            if section_name is not None:
                total += 1
                if section_name == predicted_section_name:
                    same += 1
        sample['predicted_section_names'] = predicted_section_names
    json.dump(data, open(args.output_file, 'w'))
    if total > 0:
        print (f'{same/total*100}% ({same} out of {total}) predicted labels same as original labels')

if __name__ == '__main__':
    main(args)
