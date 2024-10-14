
## Overview

We propose **DALR** (**D**ual-level **A**lignment **L**earning for multimodal sentence **R**epresentation Learning). 
To achieve cross-modal fine-grained alignment, we propose a cross-modal alignment method to mitigate the *cross-modal misalignment bias* (CMB) issue.
Furthermore, we contend that the relationships in intra-modal among sentences transcend mere positive or negative labels, encompassing a nuanced ranking structure. 
By integrating ranking distillation with global alignment learning, we effectively align intra-modal representations, alleviating the *intra-modal semantic divergence* (ISD) issue.
The following figure is an illustration of our models.

![](figure\model.png)


## Getting Started

### Download Datasets 
Run `pip install -r requirements.txt` to prepare the environment.

First you should load Flickr and MSCOCO datasets from the offical website and put them in the following format:

```bash
REPO ROOT
|
|--data    
|  |--Flickr  
|  |--MSCOCO
|  |--wiki1m_for_simcse.txt
```

**Wiki1M**
```bash
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
```

Use the script from the [SimCSE repo](https://github.com/princeton-nlp/SimCSE) to download the datasets for SentEval evaluation:

```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

You can download the model (SimCSE, DiffCSE, etc) from huggingface and put it in the `Model` folder


## Access Our Model from Google Drive
Both the model checkpoints of [flickr-bert-base](https://drive.google.com/drive/folders/1HydlrejDfI2f7KrSiiJqpLq4nimziGYY?usp=drive_link) and [coco-bert-base](https://drive.google.com/drive/folders/1MER98KbqI0gt6Mf3FwTvGterf4coUL_J?usp=drive_link) are available on Google Drive.


## Use Our model
``` python
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# Import our models. The package will take care of downloading the models from the google drives
tokenizer = AutoTokenizer.from_pretrained("Model/DALR")
model = AutoModel.from_pretrained("Model/DALR")

# Tokenize input texts
texts = [
    "There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house."
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))

```
## Evaluation

### Run Evaluation with SentEval
```bash
python eval_senteval.py \
    --model_name_or_path Model/flickr_bert_base \
    --task_set sts \
    --mode test \
```


## Train Your Own Models

In the following section, we describe how to train a MSSE model by using our code.

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.8.1
```

Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```
For unsupervised mixed setting, you can run the following command train your own models and try out different hyperparameters in it as you like
```bash
bash scripts/run_***.sh
```


## Acknowledgements

- We use the [SentEval toolkit](https://github.com/facebookresearch/SentEval) for evaluations, and we adopt the modified version of SenteEval from the [SimCSE](https://github.com/princeton-nlp/SimCSE). 
- Part of our code comes from [MCSE](https://github.com/uds-lsv/MCSE) and [KDMCSE](https://github.com/duyngtr16061999/KDMCSE).