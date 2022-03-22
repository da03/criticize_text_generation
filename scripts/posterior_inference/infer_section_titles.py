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
    for sample in tqdm(data):
        sections = sample['sections']
        predicted_section_titles = []
        for section in sections:
            inputs = tokenizer(section, padding='max_length', max_length=512, truncation=True)
            for k in inputs:
                inputs[k] = torch.LongTensor(inputs[k]).cuda().unsqueeze(0)
            outputs = model(**inputs)
            logits = outputs['logits'][0]
            label = logits.argmax(0).item()
            label = id2label[label]
            predicted_section_titles.append(label)
        sample['predicted_section_names'] = predicted_section_names
    fout.write(json.dumps(data))

if __name__ == '__main__':
    main(args)
