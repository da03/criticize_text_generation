# Model Criticism for Text Generation

Here we provide code to reproduce our results for "Critiquing Coreference Chains". We provide all training data and training scripts, as well as all pretrained models used in our paper. Our code is built on top of [Huggingface Transformers](https://github.com/huggingface/transformers/tree/de635af3f1ef740aa32f53a91473269c6435e19e).

## Instructions for Other Experiments

Instructions for "A Surprising Text Generation Failure" can be found at [../synthetic/README.md](../synthetic/README.md), instructions for "Critiquing Discourse Coherence" can be found at [../README.md](../README.md), and instructions for "Critiquing Topic Correlations" can be found at [../critique_topic_correlations/README.md](../critique_topic_correlations/README.md).

## Prerequisites

The code has been tested on Python 3.8. In addition, we need

* [Pytorch](https://pytorch.org/get-started/locally/)
* [Transformers](https://github.com/huggingface/transformers/tree/de635af3f1ef740aa32f53a9173269c6435e19e)

We use a particular version of Transformers:

```
git clone https://github.com/huggingface/transformers.git
cd transformers
export HF_DIR=$(pwd)
git checkout de635af3f1ef740aa32f53a91473269c6435e19e
pip install --editable .
```


## Datasets & Pretrained Models

For this section we only use the Wikipedia dataset, since it contains richer coreference structures. We copied over commands to train a language model and generate from it, but we recommend directly using the provided LM generations to save time.

* [data](https://drive.google.com/file/d/1stCsnajY-DB9U2-LS32tmdmsZJHtAxte/view?usp=sharing) [GPT-2 LM](https://drive.google.com/file/d/1u4-ezV74UIec6uTkMxciX8oNkHGCtq1y/view?usp=sharing) [GPT-Neo LM](https://drive.google.com/file/d/1V6S05FxaKXGff5khe87uJbsdsCVTCkGl/view?usp=sharing)
* LM generations: included in this repo in files `generation.wiki.w_title.gpt2.json`, `generation.wiki.w_title.gptneo.json`, `generation.wiki.wo_title.gpt2.json`, and `generation.wiki.wo_title.gptneo.json`.


### Data Format

Each dataset split is a list of dictionaries, where each dictionary is an example containing two fields: "sections", which is a list of section texts, and "section_names", which is a list of section titles. Model generations follow the same format, except that for the W/O Title setting "section_names" is always `None`.


## Usage

First, we setup an environment variable holding the absolute path of the current repo.

```
cd /path/to/current_repo
export WORKING_DIR=$(pwd)
```

### Train LMs (Optional)

In the paper we only presented results for the with section titles setting (W/ Title). Here we provide code for both the with section titles setting (W/ Title) and the without section titles setting (W/O Title). We need to first process data according to these settings, using the script `scripts/data/process_data_for_LMs.py`. Note that this section can be skipped if you download the pretrained language models listed above, or if you directly use the provided model generations.

```
python scripts/data/process_data_for_LMs.py --dataset_folder data/Wiki/
```

Next, we use huggingface's Transformers library for training (finetuning) language models. In particular, we use the training script `examples/legacy/run_language_modeling.py`:

```
cd $HF_DIR
cd transformers/examples/legacy
```

To train a GPT-2-based language model, we use the below command:

```
export TRAIN_FILE=${WORKING_DIR}/data/Wiki/train.w_title.txt
export TEST_FILE=${WORKING_DIR}/data/Wiki/val.w_title.txt
export B=8
export A=1
export epochs=30
export model=gpt2
python run_language_modeling.py \
       --per_device_train_batch_size=${B} \
       --gradient_accumulation_steps=${A} \
       --output_dir=${WORKING_DIR}/language_model_checkpoints/Wiki/GPT-2 \
       --model_type=$model \
       --model_name_or_path=$model \
       --do_train \
       --do_eval \
       --train_data_file=$TRAIN_FILE \
       --eval_data_file=$TEST_FILE --overwrite_output_dir --save_total_limit=5 \
       --learning_rate=5e-5 --num_train_epochs=${epochs} --load_best_model_at_end=True \
       --evaluation_strategy=epoch --save_strategy=epoch > ${WORKING_DIR}/log.pubmed.trainLM.w_title.gpt2 2>&1&
```

Note that above we only showed the training commands for the W/ Title setting. We use the same settings for the W/O Title setting except for the training and validation files.

To train a GPT-Neo-based language model, we use the below command:

```
export TRAIN_FILE=${WORKING_DIR}/data/Wiki/train.w_title.txt
export TEST_FILE=${WORKING_DIR}/data/Wiki/val.w_title.txt
export B=4
export A=2
export epochs=30
export model=EleutherAI/gpt-neo-125M
python run_language_modeling.py \
       --per_device_train_batch_size=${B} \
       --gradient_accumulation_steps=${A} \
       --output_dir=${WORKING_DIR}/language_model_checkpoints/Wiki/GPT-Neo \
       --model_type=$model \
       --model_name_or_path=$model \
       --do_train \
       --do_eval \
       --train_data_file=$TRAIN_FILE \
       --eval_data_file=$TEST_FILE --overwrite_output_dir --save_total_limit=5 \
       --learning_rate=5e-5 --num_train_epochs=${epochs} --load_best_model_at_end=True \
       --evaluation_strategy=epoch --save_strategy=epoch > ${WORKING_DIR}/log.pubmed.trainLM.w_title.gptneo 2>&1&

```

Training takes a day on a single Nvidia A100 GPU.

### Generate from LMs (Optional)

Note that this section can also be skipped if you directly download the model generations from above.

```
cd $WORKING_DIR
```

To generate from a trained language model, for the W/ Title setting:

```
 python scripts/generate/sample_LM.py \
        --language_model_checkpoint language_model_checkpoints/Wiki/GPT-2/With_Title \
        --output_file generation.pubmed.w_title.gpt2.json \
        --with_title \
        --num_samples 100
```

For the W/O Title setting:

```
 python scripts/generate/sample_LM.py \
        --language_model_checkpoint language_model_checkpoints/Wiki/GPT-2/Without_Title \
        --output_file generation.pubmed.wo_title.gpt2.json \
        --num_samples 100
```

Note that in the paper we used 10k samples, which requires setting `--num_samples` to 10000 (it takes much longer to generate 10k samples so we used 100 in the above example commands).

### Fit Critic Generative Processes (Optional)

We need to fit the critic distribution $P_c(z)$ (this step is optional if you use the pretrained critic models):

```
python scripts/criticize/fit_critic.py --dataset_folder data/PubMed/ --output_folder critic_checkpoints/PubMed/
```

### Train Posterior Inferencers (Optional)

We use huggingface transformer's script `examples/pytorch/text-classification/run_glue.py` to train a posterior inferencer. First, we need to prepare data according to its expected format, using the script `scripts/data/process_data_for_posterior_inferencers.py`. Note that this section can be skipped if you download the pretrained posterior inferencers listed in the beginning of this document.

```
python scripts/data/process_data_for_posterior_inferencers.py --dataset_folder data/PubMed/
```

Next, we can train a posterior inferencer (starting from the root directory of huggingface's Transformers):

```
cd $HF_DIR
cd examples/pytorch/text-classification
```

```
python run_glue.py \
    --model_name_or_path=bert-base-cased \
    --do_train \
    --do_eval \
    --train_file=${WORKING_DIR}/data/PubMed/train.posterior_inferencer.json \
    --validation_file=${WORKING_DIR}/data/PubMed/val.posterior_inferencer.json \
    --max_seq_length=512 \
    --per_gpu_train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=${WORKING_DIR}/posterior_inferencer_checkpoints/PubMed \
    --save_total_limit=5 \
    --overwrite_output_dir > ${WORKING_DIR}/log.pubmed.trainPosteriorInferencer 2>&1&
```

### Posterior Inference

The goal of posterior inference is to infer the section titles z conditioned on section text x. We use a BERT-based classifier and use the MAP value of z instead of maintaining a full distribution over z:

```
python scripts/posterior_inference/infer_section_titles.py \
       --posterior_inferencer_checkpoint posterior_inferencer_checkpoints/PubMed \
       --input_file generation.pubmed.w_title.gpt2.json \
       --output_file predicted_z.generation.pubmed.w_title.gpt2.json
```

### Model Criticism in Latent Space

Now we are ready to criticize in the latent space.

```
python scripts/criticize/criticize.py \
       --critic critic_checkpoints/PubMed/ \
       --input_file predicted_z.generation.pubmed.w_title.gpt2.json
```

The output will contain the latent PPL for the LM generations.


## Acknowledgements

The datasets PubMed and ArXiv are adapted from [Cohan et. al. 2018](https://aclanthology.org/N18-2097/). Wiki is processed based on the English Wikipedia [dumped on Dec 1, 2021](https://dumps.wikimedia.org/enwiki/20211201/).