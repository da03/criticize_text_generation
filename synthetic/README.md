# Model Criticism for Long-Form Text Generation

Here we provide code to reproduce our results for "A Surprising Text Generation Failure". We provide all training data and training scripts, as well as all pretrained models used in our paper. Our code is built on top of [fairseq](https://github.com/pytorch/fairseq) and [pytorch-struct](https://github.com/harvardnlp/pytorch-struct/tree/e51fecc1473925e4c44de135c4a3240fcb20fa40).

## Instructions for Other Experiments

Instructions for "Critiquing Discourse Coherence" can be found at [../README.md](../README.md), and instructions for "Critiquing Discourse Coherence" can be found at [../critique_topic_correlations/README.md](../critique_topic_correlations/README.md).

## Prerequisites

The code has been tested on Python 3.8. In addition, we need

* [Pytorch](https://pytorch.org/get-started/locally/)
* [fairseq](https://github.com/huggingface/transformers/tree/0e608fdba6cd27bb2aa917e369a0f49a2c55cb1e)
* [pytorch-struct](https://github.com/harvardnlp/pytorch-struct/tree/e51fecc1473925e4c44de135c4a3240fcb20fa40)

We use a particular version of fairseq and apply a patch:

```
git clone https://github.com/pytorch/fairseq.git
cd fairseq
export FAIRSEQ_DIR=$(pwd)
git checkout 0e608fdba6cd27bb2aa917e369a0f49a2c55cb1e
git apply /path/to/current_repo/synthetic/scripts/utils/fix_lm_sampling.patch
pip install --editable .
```

For HSMM language model training and sampling, we use a particular version of pytorch-struct and also apply a patch for more memory-efficient partition computation.

```
git clone git@github.com:harvardnlp/pytorch-struct.git
cd pytorch-struct
git checkout e51fecc1473925e4c44de135c4a3240fcb20fa40
git apply /path/to/current_repo/synthetic/scripts/utils/memory_efficient_semimarkov.patch
pip install --editable .
```

## Dataset & Pretrained Models

* [data](https://drive.google.com/file/d/19jqKf6oVgVuwHkFWrUmqjaYT_KywjKCo/view?usp=sharing) [Transformer LM](https://drive.google.com/file/d/1ayfQXTY4h4wcNxFNbnyYScB6uOlMm7yr/view?usp=sharing) [HSMM LM](https://drive.google.com/file/d/14g-EVW3yQmbg_DyA57v4vpmWPauQ8AvX/view?usp=sharing)


### Data Format

Each dataset split is a list of dictionaries, where each dictionary is an example containing two fields: "sections", which is a list of section texts, and "section_names", which is a list of section titles.

## Usage

First, we setup an environment variable holding the absolute path of the current repo.

```
export WORKING_DIR=$(pwd)
```

### Generate Dataset (Optional)

The below commands generate the dataset. This step is optional if you download the pre-generated data from the link above.

```
cd ${WORKING_DIR}/synthetic
```

```
python scripts/data/generate_dataset.py --dataset_folder data
```

The generated word-level data is stored in `data/train.x`, `data/val.x`, and `data/test.x`. Note that due to PyTorch's [reproducibility issues](https://pytorch.org/docs/stable/notes/randomness.html), the generated dataset might vary very slightly across different PyTorch versions. The data we use is generated with PyTorch 1.4.0.

### Train LMs (Optional)

#### Train Transformer LM (Optional)

We use `fairseq` for training a transformer language model on this dataset. Note that this section can be skipped if you download the pretrained transformer language model listed above.

```
cd ${FAIRSEQ_DIR}
```

First, we need to process data into binary format.

```
fairseq-preprocess \
    --only-source \
    --trainpref ${WORKING_DIR}/synthetic/data/train.x \
    --validpref ${WORKING_DIR}/synthetic/data/val.x \
    --testpref ${WORKING_DIR}/synthetic/data/test.x \
    --destdir ${WORKING_DIR}/synthetic/data/data-bin \
    --workers 20 \
    --padding-factor 1
```

```
fairseq-train --task language_modeling \
  ${WORKING_DIR}/synthetic/data/data-bin \
  --save-dir ${WORKING_DIR}/synthetic/language_model_checkpoints/transformer \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.3 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 4098 --sample-break-mode eos \
  --max-tokens 4096 \
  --max-update 120000 --no-epoch-checkpoints --seed 1234
```

#### Train HSMM LM (Optional)

This section shows how to train an HSMM language model. Note that this section is optional if you download the pretrained HSMM language model from above.

The HSMM language model is parameterized by three matrices: a transition matrix that specifies the transition probability from a previous state to the next state; a length emission matrix that specifies the distribution over the number of tokens in the subsequence to generate given a state; an emission matrix that specifies the distribution over the subsequence of tokens given a state and the number of tokens in the subsequence. Note that to parameterize the emission matrix, we cannot use the subsequence vocabulary since that would be cheating. Instead, we take the most common 250k n-grams in the training dataset and use them as the possible subsequences to emit:

```
python scripts/train_hsmm/build_subseq_vocab.py \
    --dataset_folder data \
    --output_folder language_model_checkpoints/hsmm \
    --subseq_vocab_size 250000 \
    --subseq_min_len 1 --subseq_max_len 11
```

We'll use the generated subsequence vocabulary to convert ngrams in the training datasets into subsequence ids:

```
python scripts/train_hsmm/convert_data_to_subseq_ids.py \
    --dataset_folder data \
    --vocab_folder language_model_checkpoints/hsmm
```

Now we are ready to train the HSMM. HSMM training consists of two phases: in the first phase, we fix the transition matrix and the length emission matrix, and only learn the emission matrix. In the second phase, we train everything. The first training phase is necessary to avoid under-utilization of states.

For the first training phase:

```
python scripts/train_hsmm/train.py \
    --dataset_folder data \
    --vocab_folder language_model_checkpoints/hsmm \
    --checkpoint_folder language_model_checkpoints/hsmm \
    --Z 800 \
    --fix_z_transitions \
    --fix_n_emissions \
    --batch_size 8 \
    --epochs 1 \
    --lr 1e-1 > log.train.HSMM.phase1 2>&1&
```

For the second training phase:

```
python scripts/train/train.py \
    --dataset_folder data \
    --vocab_folder language_model_checkpoints/hsmm \
    --checkpoint_folder language_model_checkpoints/hsmm \
    --Z 800 \
    --batch_size 8 \
    --epochs 10 \
    --lr 3e-1 \
    --train_from language_model_checkpoints/hsmm/checkpoint_best.pt > log.train.HSMM.phase2 2>&1&
```

Note that due to the small batch size used (since the emission parameters consume a lot of memory. Besides, HSMM inference is memory-intensive), training is extremely slow and can take a week on a single Nvidia A100 GPU.


### Generate from LMs

#### Generate from Transformer LM

```
cd ${FAIRSEQ_DIR}
```

To generate from a trained language model, we use `fairseq-generate` (don't forget to apply the patch in the installation part since otherwise fairseq would simply copy the original sentences):

```
fairseq-generate ${WORKING_DIR}/synthetic/data/data-bin \
    --path ${WORKING_DIR}/synthetic/language_model_checkpoints/transformer/checkpoint_best.pt \
    --batch-size 128  --max-len-a 0 --max-len-b 4096 --sampling --beam 1 --nbest 1 --sample-break-mode eos \
    --task language_modeling | tee ${WORKING_DIR}/synthetic/log.generate.transformer
```

```
cd ${WORKING_DIR}/synthetic
grep ^H log.generate.transformer | cut -f3- > generation.transformer.txt
```

#### Generate from HSMM LM

To generate from a trained HSMM language model:

```
python scripts/train_hsmm/generate.py \
    --checkpoint_path language_model_checkpoints/hsmm/checkpoint_best.pt \
    --vocab_folder language_model_checkpoints/hsmm/ \
    --num_samples 6400 \
    --output_file generation.HSMM.txt
```

The generations will be written to `generation.HSMM.txt`.


### Posterior Inference

The goal of posterior inference is to infer the latent states z conditioned on observed x. By design the posterior distribution is a delta distribution so we can find a deterministic mapping from x to z.

```
cd ${WORKING_DIR}/synthetic
python scripts/posterior_inference/infer_z.py \
       --dataset_folder data \
       --input_file generation.transformer.txt \
       --output_file predicted_z.generation.transformer.txt
```

```
python scripts/posterior_inference/infer_z.py \
       --dataset_folder data \
       --input_file generation.HSMM.txt \
       --output_file predicted_z.generation.HSMM.txt
```

### Model Criticism in Latent Space

Now we are ready to criticize in the latent space.

```
cd ${WORKING_DIR}/synthetic
python scripts/criticize/criticize.py \
       --dataset_folder data/ \
       --input_file_z predicted_z.generation.transformer.txt
```

```
python scripts/criticize/criticize.py \
       --dataset_folder data/ \
       --input_file_z predicted_z.generation.HSMM.txt
```

The outputs will contain the latent PPL for both real data and LM generations.
