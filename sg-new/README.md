# SG-New

This is our reboot of SG using the Huggingface transformers library 


## Environment

Recommend to create a new conda environment called huggingface


```bash

conda create --name huggingface tensorflow==2.0 pytorch
conda activate huggingface
pip install transformers
```


You may need to do a source install of huggingface:

```bash
conda activate huggingface
git clone git@github.com:huggingface/transformers.git
cd transformers
pip install . 
```

## Dataset

We will be using the  GLUE STS-B benchmark and dataset for test (for now).

You should first make sure that you have hte dataset cloned and ready:

```bash
cd $SCAIA_HOME
git clone git@github.com:ScaiaAi/datasets.git
cd datasets/glue/
./download_glue_data.sh
```

## Example

I have installed the huggingface example in the [example](./example) folder.

This is not our code, but it is included here as a refernece.

To run:

```bash
export SCAIA_HOME=/path/to/your/scaia/home
cd $SCAIA_HOME/example
cd example
./run_example.sh

```


## Code

The code is as follows:


### Data Preparation
 * [prepare_data.py](prepare_data.py) : prepare data
 * [tokenize.py](tokenize.py) : tokenize data using BERT (or other) tokenizer

### Training

 * [train.py](train.py)  : train model
 * [finetune.py](finetune.py) : finetune transformer
 * [evaluate.py](evaluate.py) : perform evaluation
 
### Runtime

 * [transform.py](transform.py) : Transforms documetns using trained embedding model
 * [similarity.py](similarity.py) : Calculates Similarity of Documents
 * [find_similar.py](find_similar.py) : This will find similar documents
