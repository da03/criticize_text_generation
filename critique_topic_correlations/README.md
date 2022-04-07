# Model Criticism for Text Generation

Here we provide code to reproduce our results for "Critiquing Topic Correlations". We provide all training data and training scripts, as well as all pretrained models used in our paper. Our code is built on top of [Huggingface Transformers](https://github.com/huggingface/transformers/tree/de635af3f1ef740aa32f53a91473269c6435e19e), David Blei and John Lafferty's [CTM](http://www.cs.columbia.edu/~blei/ctm-c/).

## Instructions for Other Experiments

Instructions for "A Surprising Text Generation Failure" can be found at [synthetic/README.md](/synthetic/README.md), and instructions for "Critiquing Discourse Coherence" can be found at [README.md](/README.md).

## Prerequisites

* [Pytorch](https://pytorch.org/get-started/locally/)
* [Transformers](https://github.com/huggingface/transformers/tree/de635af3f1ef740aa32f53a9173269c6435e19e)
* [CTM](http://www.cs.columbia.edu/~blei/ctm-c/)
* seaborn (`pip install seaborn`)

We use CTM for learning the critic generative process and for performing posterior inference.

```
wget http://www.cs.columbia.edu/~blei/ctm-c/ctm-dist.tgz
tar zxf ctm-dist.tgz
cd ctm-dist
make
export CTM_DIR=$(pwd)
```

We use a particular version of Transformers:

```
git clone https://github.com/huggingface/transformers.git
cd transformers
export HF_DIR=$(pwd)
git checkout de635af3f1ef740aa32f53a91473269c6435e19e
pip install --editable .
```


## Datasets & Pretrained Models

* PubMed: [data](https://drive.google.com/file/d/1qIIJBc6JhxSipsdz7X9XgEPsBPrB5PDS/view?usp=sharing) [GPT-2 LM](https://drive.google.com/file/d/1MQleq0eBW3vxQU0fd_xTIAMPuSakL4vu/view?usp=sharing) [GPT-Neo LM](https://drive.google.com/file/d/13QbzOCnpjQuhoZ4ZSFMz87FmQ7CLPPCf/view?usp=sharing) [Critic](https://drive.google.com/file/d/1SMnYe7qfpSOdhACK3xikUhnZYyOrgJxk/view?usp=sharing)
* ArXiv: [data](https://drive.google.com/file/d/1Ujn84S-37r0I1z7Uhq-8aCvo5RIaSobs/view?usp=sharing) [GPT-2 LM](https://drive.google.com/file/d/17mNJoUwROEo0OWSF2llYXLsT4_1SnCrQ/view?usp=sharing) [GPT-Neo LM](https://drive.google.com/file/d/1bbofIpumvf_1StMf59pUR9-Lvln6qT1b/view?usp=sharing) [Critic](https://drive.google.com/file/d/1ixP-QY_Oe7t9gDz2Z7OJv-u9H9LQE1mh/view?usp=sharing)
* Wiki: [data](https://drive.google.com/file/d/1stCsnajY-DB9U2-LS32tmdmsZJHtAxte/view?usp=sharing) [GPT-2 LM](https://drive.google.com/file/d/1u4-ezV74UIec6uTkMxciX8oNkHGCtq1y/view?usp=sharing) [GPT-Neo LM](https://drive.google.com/file/d/1V6S05FxaKXGff5khe87uJbsdsCVTCkGl/view?usp=sharing) [Critic](https://drive.google.com/file/d/1Bg9YnAN1JHjgj6jYdTSh5FNjtCgzydMo/view?usp=sharing)


### Data Format

Each dataset split is a list of dictionaries, where each dictionary is an example containing two fields: "sections", which is a list of section texts, and "section_names", which is a list of section titles.


## Usage

First, we setup an environment variable holding the absolute path of the current repo.

```
cd /path/to/current_repo
export WORKING_DIR=$(pwd)
```

### Train LMs (Optional)

In the paper we considered two different data settings: with section titles (W/ Title) and without section titles (W/O Title). We need to first process data according to these settings, using the script `scripts/data/process_data_for_LMs.py`. Note that this section can be skipped if you download the pretrained language models listed above.

```
python scripts/data/process_data_for_LMs.py --dataset_folder ${WORKING_DIR}/data/PubMed/
```

Next, we use huggingface's Transformers library for training (finetuning) language models. In particular, we use the training script `examples/legacy/run_language_modeling.py`:

```
cd transformers/examples/legacy
```

We will use the dataset PubMed to illustrate training (the commands for other datasets are the same). To train a GPT-2-based language model, we use the below command:

```
export TRAIN_FILE=${WORKING_DIR}/data/PubMed/train.w_title.txt
export TEST_FILE=${WORKING_DIR}/data/PubMed/val.w_title.txt
export B=8
export A=1
export epochs=30
export model=gpt2
python run_language_modeling.py \
       --per_device_train_batch_size=${B} \
       --gradient_accumulation_steps=${A} \
       --output_dir=${WORKING_DIR}/language_model_checkpoints/PubMed/GPT-2 \
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
export TRAIN_FILE=${WORKING_DIR}/data/PubMed/train.w_title.txt
export TEST_FILE=${WORKING_DIR}/data/PubMed/val.w_title.txt
export B=4
export A=2
export epochs=30
export model=EleutherAI/gpt-neo-125M
python run_language_modeling.py \
       --per_device_train_batch_size=${B} \
       --gradient_accumulation_steps=${A} \
       --output_dir=${WORKING_DIR}/language_model_checkpoints/PubMed/GPT-Neo \
       --model_type=$model \
       --model_name_or_path=$model \
       --do_train \
       --do_eval \
       --train_data_file=$TRAIN_FILE \
       --eval_data_file=$TEST_FILE --overwrite_output_dir --save_total_limit=5 \
       --learning_rate=5e-5 --num_train_epochs=${epochs} --load_best_model_at_end=True \
       --evaluation_strategy=epoch --save_strategy=epoch > ${WORKING_DIR}/log.pubmed.trainLM.w_title.gptneo 2>&1&

```

Training takes a few hours to a day on a single Nvidia A100 GPU, depending on the dataset.

### Generate from LMs

```
cd $WORKING_DIR
```

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


### Process Data for Critic Training/Inference

We use David Blei and John Lafferty's correlated topic model implementation [CTM](http://www.cs.columbia.edu/~blei/ctm-c/) for learning a critic generative process and for posterior inference. We need to prepare data according to its expected format first.

```
cd ${WORKING_DIR}/critique_topic_correlations
```

First, we need to build the vocabulary. By default we consider word types that appear in more than 50% of documents as stopwords, and we also remove words that appear fewer than 5 times.

```
python scripts/data/build_vocab_for_critics.py --dataset_folder ${WORKING_DIR}/data/PubMed --compatible_with_checkpoints
```

Based on the constructed vocabulary, we can convert words to word ids and generate input files to CTM.

```
python scripts/data/process_data_for_critics.py --vocab_file ${WORKING_DIR}/data/PubMed/train.json.CTM.vocab --data_file ${WORKING_DIR}/data/PubMed/train.json 
python scripts/data/process_data_for_critics.py --vocab_file ${WORKING_DIR}/data/PubMed/train.json.CTM.vocab --data_file ${WORKING_DIR}/data/PubMed/val.json 
python scripts/data/process_data_for_critics.py --vocab_file ${WORKING_DIR}/data/PubMed/train.json.CTM.vocab --data_file ${WORKING_DIR}/data/PubMed/test.json 
```

To criticize language model generations, we need to process them similarly. For example, to criticize `generation.pubmed.wo_title.gpt2.json`, we need to

```
python scripts/data/process_data_for_critics.py --vocab_file ${WORKING_DIR}/data/PubMed/train.json.CTM.vocab --data_file ${WORKING_DIR}/generation.pubmed.wo_title.gpt2.json
```


### Fit Critic Generative Processes (Optional)
python ctm-topics.py arxiv100Fix/final-log-beta.dat /n/holyscratch01/rush_lab/Users/yuntian/hierarchy/arxiv-final-2k-dataset/train_flat_nonewline.txt.addspecial.filter.ctm.vocab arxiv_topics.txt 25

### Posterior Inference

We use CTM for posterior inference.

```
cd ${CTM_DIR}
```

First, we perform posterior inference on the ground truth data.

```
mkdir -p ${WORKING_DIR}/critique_topic_correlations/results/PubMed
./ctm inf ${WORKING_DIR}/data/PubMed/test.json.CTM.id ${WORKING_DIR}/critique_topic_correlations/critic_checkpoints/PubMed/final ${WORKING_DIR}/critique_topic_correlations/results/PubMed/test inf-settings.txt
```

Next, we perform posterior inference on language model generations.

```
./ctm inf ${WORKING_DIR}/generation.pubmed.wo_title.gpt2.json.CTM.id ${WORKING_DIR}/critique_topic_correlations/critic_checkpoints/PubMed/final ${WORKING_DIR}/critique_topic_correlations/results/PubMed/wo_title.gpt2 inf-settings.txt
```

The inference results will be written to `results/PubMed/`.


### Model Criticism in Latent Space

Now we are ready to criticize in the latent space.

```
cd ${WORKING_DIR}/critique_topic_correlations
```

 python evaluate_llh_lambda_compare_baseline.py 100 pubmed100FixRerun/final-mu.dat pubmed100FixRerun/final-inv-cov.dat  pubmed100FixRerun/final-cov.dat  pubmed100Fix/test-lambda.dat pubmed100Fix/gpt2-lambda.dat pubmed100Fix/gpt2-arxiv-lambda.dat cov_pubmed_arxivbaseline

```
python scripts/criticize/criticize.py \
       --critic critic_checkpoints/PubMed/ \
       --real_data_lambda results/PubMed/test-lambda.dat \
       --LM_generations_lambda results/PubMed/wo_title.gpt2-lambda.dat \
       --visualize_cov_path results/PubMed/cov-test-wo_title.gpt2.png \
       --hierarchical_clustering
```

The latent NLLs will be printed out, and the visualization of covariance matrices will be stored in `results/PubMed/cov-test-wo_title.gpt2.png`.


## Acknowledgements

The datasets PubMed and ArXiv are adapted from [Cohan et. al. 2018](https://aclanthology.org/N18-2097/). Wiki is processed based on the English Wikipedia [dumped on Dec 1, 2021](https://dumps.wikimedia.org/enwiki/20211201/).
