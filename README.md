# Model Criticism for Text Generation

Here we provide code to reproduce our results. We provide all training data and training scripts, as well as all pretrained models used in our paper. Our code is built on top of [🤗 Transformers] (https://github.com/huggingface/transformers/tree/de635af3f1ef740aa32f53a91473269c6435e19e), [fairseq](https://github.com/pytorch/fairseq), and [pytorch-struct](https://github.com/harvardnlp/pytorch-struct).

## Prerequisites

* [Pytorch](https://pytorch.org/get-started/locally/)
* [🤗 Transformer](hhttps://github.com/huggingface/transformers/tree/de635af3f1ef740aa32f53a9173269c6435e19e)


## Datasets & Pretrained Models

* PubMed: [data](https://drive.google.com/file/d/1qIIJBc6JhxSipsdz7X9XgEPsBPrB5PDS/view?usp=sharing) [GPT-2 LM](https://drive.google.com/file/d/1MQleq0eBW3vxQU0fd_xTIAMPuSakL4vu/view?usp=sharing) [GPT-Neo LM](https://drive.google.com/file/d/13QbzOCnpjQuhoZ4ZSFMz87FmQ7CLPPCf/view?usp=sharing) [Posterior Inferencer](https://drive.google.com/file/d/1-BwbijwD6nIKOMOkV3FOWqRHdHKIuaqY/view?usp=sharing)
* ArXiv: [data](https://drive.google.com/file/d/1Ujn84S-37r0I1z7Uhq-8aCvo5RIaSobs/view?usp=sharing) [GPT-2 LM](https://drive.google.com/file/d/17mNJoUwROEo0OWSF2llYXLsT4_1SnCrQ/view?usp=sharing) [GPT-Neo LM](https://drive.google.com/file/d/1bbofIpumvf_1StMf59pUR9-Lvln6qT1b/view?usp=sharing) [Posterior Inferencer](https://drive.google.com/file/d/1mD98cWmpD2ja4H3gIQD_2BT1UzKor74U/view?usp=sharing)
* Wiki: [data](https://drive.google.com/file/d/1stCsnajY-DB9U2-LS32tmdmsZJHtAxte/view?usp=sharing) [GPT-2 LM](https://drive.google.com/file/d/1u4-ezV74UIec6uTkMxciX8oNkHGCtq1y/view?usp=sharing) [GPT-Neo LM](https://drive.google.com/file/d/1V6S05FxaKXGff5khe87uJbsdsCVTCkGl/view?usp=sharing) [Posterior Inferencer](https://drive.google.com/file/d/118DJq-C5BMP83tuoZSQPc337G7_n4WDR/view?usp=sharing)


## Usage

### Train LMs

In the paper we considered two different data settings: with section titles (W/ Title) and without section titles (W/O Title). We need to first process data according to these settings.

### Generate from LMs
