import os
import sys
import json
import argparse
import tqdm

import torch
from transformers import AutoModelForCausalLM, GPT2Tokenizer


parser = argparse.ArgumentParser(description='Sample from the language models.')
parser.add_argument('--language_model_checkpoint', type=str, required=True,
                    help='Language model checkpoint. Should be a folder containing pytorch_model.bin and other configuration files.')
parser.add_argument('--with_title', action='store_true',
                    help='Whether or not the setting is W/ Title.')
parser.add_argument('--output_file', type=str, required=True,
                    help='Output filename.')
parser.add_argument('--num_samples', type=int, default=10000,
                    help='Number of articles to sample.')
parser.add_argument('--max_length', type=int, default=20000,
                    help='Maximum possible generation length. Generation will stop after generating this many of tokens, if no eos token is generated.')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed.')
args = parser.parse_args()


def main(args):
    # The tokenizer is GPT-2 for both GPT-2 and GPT-Neo LMs
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    eos_token = """<|endoftext|>"""
    eos_id = tokenizer.encode(eos_token, add_special_tokens=False, return_tensors='pt').item()

    #import pdb; pdb.set_trace()
    assert eos_id == tokenizer.eos_token_id

    # Load LM
    model = AutoModelForCausalLM.from_pretrained(args.language_model_checkpoint, pad_token_id=tokenizer.eos_token_id)
    model.cuda()
    model.eval()
    
    torch.manual_seed(args.seed)
    
    block_size = model.config.max_position_embeddings
    max_length = args.max_length
    
    assert not os.path.exists(args.output_file), f'{args.output_file} already exists!'
    num_generated = 0
    samples = []
    while num_generated < args.num_samples:
        prefix = """<|endoftext|>\n<|endoftext|>"""
        input_ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt').cuda()
        if max_length <= block_size:
            sample_output = model.generate(input_ids, max_length=max_length, do_sample=True, top_p=1.0)
        else:
            sample_output = model.generate(input_ids, max_length=block_size, do_sample=True, top_p=1.0)
            num_left = max_length - block_size
            for i in range(num_left):
                context = sample_output[:, -(block_size-1):]
                if context[:, -1].item() == eos_id:
                    break
                step_output = model.generate(context, max_length=block_size, do_sample=True, top_p=1.0)[:, context.size(-1):]
                sample_output = torch.cat((sample_output, step_output), dim=-1)
    
        text = tokenizer.decode(sample_output[0], skip_special_tokens=False)
        text = text.replace('<|endoftext|>', '').strip()
        invalid = False
        if len(text) == 0:
            invalid = True
        sections_all = text.split('\n\n')
        if args.with_title:
            section_names = []
            sections = []
            for section in sections_all:
                try:
                    section_name, section_text = section.strip().split(':', 1)
                    section_text = section_text.strip()
                    if len(section_text) == 0:
                        invalid = True
                except Exception as e:
                    invalid = True
                section_names.append(section_name)
                sections.append(section_text)
        else:
            sections = sections_all
            section_names = None
        if invalid:
            print ('invalid generation, skipping')
            continue
        num_generated += 1
        print (f'{num_generated} out of {args.num_samples}')
        sys.stdout.flush()
        sample = {'section_names': section_names, 'sections': sections}
        samples.append(sample)
    json.dump(samples, open(args.output_file, 'w'))

if __name__ == '__main__':
    main(args)
