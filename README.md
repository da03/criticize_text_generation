# Model Criticism for Text Generation

Here we provide code to reproduce our results. We provide all training data and training scripts, as well as all pretrained models used in our paper. Our code is built on top of [Transformers](https://github.com/huggingface/transformers/tree/de635af3f1ef740aa32f53a91473269c6435e19e), [fairseq](https://github.com/pytorch/fairseq), and [pytorch-struct](https://github.com/harvardnlp/pytorch-struct).

## Prerequisites

* [Pytorch](https://pytorch.org/get-started/locally/)
* [Transformer](https://github.com/huggingface/transformers/tree/de635af3f1ef740aa32f53a9173269c6435e19e)


## Datasets & Pretrained Models

* PubMed: [data](https://drive.google.com/file/d/1qIIJBc6JhxSipsdz7X9XgEPsBPrB5PDS/view?usp=sharing) [GPT-2 LM](https://drive.google.com/file/d/1MQleq0eBW3vxQU0fd_xTIAMPuSakL4vu/view?usp=sharing) [GPT-Neo LM](https://drive.google.com/file/d/13QbzOCnpjQuhoZ4ZSFMz87FmQ7CLPPCf/view?usp=sharing) [Posterior Inferencer](https://drive.google.com/file/d/1-BwbijwD6nIKOMOkV3FOWqRHdHKIuaqY/view?usp=sharing)
* ArXiv: [data](https://drive.google.com/file/d/1Ujn84S-37r0I1z7Uhq-8aCvo5RIaSobs/view?usp=sharing) [GPT-2 LM](https://drive.google.com/file/d/17mNJoUwROEo0OWSF2llYXLsT4_1SnCrQ/view?usp=sharing) [GPT-Neo LM](https://drive.google.com/file/d/1bbofIpumvf_1StMf59pUR9-Lvln6qT1b/view?usp=sharing) [Posterior Inferencer](https://drive.google.com/file/d/1mD98cWmpD2ja4H3gIQD_2BT1UzKor74U/view?usp=sharing)
* Wiki: [data](https://drive.google.com/file/d/1stCsnajY-DB9U2-LS32tmdmsZJHtAxte/view?usp=sharing) [GPT-2 LM](https://drive.google.com/file/d/1u4-ezV74UIec6uTkMxciX8oNkHGCtq1y/view?usp=sharing) [GPT-Neo LM](https://drive.google.com/file/d/1V6S05FxaKXGff5khe87uJbsdsCVTCkGl/view?usp=sharing) [Posterior Inferencer](https://drive.google.com/file/d/118DJq-C5BMP83tuoZSQPc337G7_n4WDR/view?usp=sharing)

The datasets PubMed and ArXiv are adapted from [Cohan et. al. 2018](https://aclanthology.org/N18-2097/). Wiki is processed based on the English Wikipedia [dumped on Dec 1, 2021](https://dumps.wikimedia.org/enwiki/20211201/).

### Data Format

Each dataset split is a list of dictionaries, where each dictionary is an example containing two fields: "sections", which is a list of section texts, and "section_names", which is a list of section titles.


## Usage

### Train LMs (Optional)

In the paper we considered two different data settings: with section titles (W/ Title) and without section titles (W/O Title). We need to first process data according to these settings, using the script `scripts/data/process_data.py`. Note that this section can be skipped if you download the pretrained language models listed above.

```
python scripts/data/process_data.py --dataset_folder data/PubMed/
python scripts/data/process_data.py --dataset_folder data/ArXiv/
python scripts/data/process_data.py --dataset_folder data/Wiki/
```

Next, we use huggingface's Transformer library for training (finetuning) language models:

```
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout de635af3f1ef740aa32f53a91473269c6435e19e
pip install --editable .
```

In particular, we use the training script in `examples/legacy`:

```
cd examples/legacy
```

We will use the dataset PubMed to illustrate training (the commands for other datasets are the same). To train a GPT-2-based language model, we use the below command:

```
export TRAIN_FILE=data/PubMed/train.w_title.txt
export TEST_FILE=data/PubMed/val.w_title.txt
export B=8
export A=1
export epochs=30
export model=gpt2
python run_language_modeling.py \
       --per_device_train_batch_size=${B} \
       --gradient_accumulation_steps=${A} \
       --output_dir=language_model_checkpoints/PubMed/GPT-2 \
       --model_type=$model \
       --model_name_or_path=$model \
       --do_train \
       --do_eval \
       --train_data_file=$TRAIN_FILE \
       --eval_data_file=$TEST_FILE --overwrite_output_dir --save_total_limit=5 \
       --learning_rate=5e-5 --num_train_epochs=${epochs} --load_best_model_at_end=True \
       --evaluation_strategy=epoch --save_strategy=epoch > log.pubmed.trainLM.w_title.gpt2 2>&1&
```

Note that above we only showed the training commands for the W/ Title setting. We use the same settings for the W/O Title setting except for the training and validation files.

To train a GPT-Neo-based language model, we use the below command:

```
export TRAIN_FILE=data/PubMed/train.w_title.txt
export TEST_FILE=data/PubMed/val.w_title.txt
export B=4
export A=2
export epochs=30
export model=EleutherAI/gpt-neo-125M
python run_language_modeling.py \
       --per_device_train_batch_size=${B} \
       --gradient_accumulation_steps=${A} \
       --output_dir=language_model_checkpoints/PubMed/GPT-Neo \
       --model_type=$model \
       --model_name_or_path=$model \
       --do_train \
       --do_eval \
       --train_data_file=$TRAIN_FILE \
       --eval_data_file=$TEST_FILE --overwrite_output_dir --save_total_limit=5 \
       --learning_rate=5e-5 --num_train_epochs=${epochs} --load_best_model_at_end=True \
       --evaluation_strategy=epoch --save_strategy=epoch > log.pubmed.trainLM.w_title.gptneo 2>&1&

```

Training takes a few hours to a day on a single Nvidia A100 GPU, depending on the dataset.

### Generate from LMs

To generate from a trained language model, for the W/ Title setting:

```
 python scripts/generate/sample_LM.py \
        --language_model_checkpoint language_model_checkpoints/PubMed/GPT-2/With_Title \
        --output_file generation.pubmed.w_title.gpt2.json \
        --with_title \
        --num_samples 100
```

For the W/O Title setting:

```
 python scripts/generate/sample_LM.py \
        --language_model_checkpoint language_model_checkpoints/PubMed/GPT-2/Without_Title \
        --output_file generation.pubmed.wo_title.gpt2.json \
        --num_samples 100
```

Note that in the paper we used 10k samples, which requires setting `--num_samples` to 10000 (it takes much longer to generate 10k samples so we used 100 in the above example commands).


### Posterior Inference

The goal of posterior inference is to infer the section titles z conditioned on section text x. We use a BERT-based classifier and use the MAP value of z instead of maintaining a full distribution over z.
