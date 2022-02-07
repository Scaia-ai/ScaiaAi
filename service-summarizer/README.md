# Summarizer Service


## Environment

Create a conda enviornment for scaia

```bash
conda create --name scaia python=3.8 flask
conda activate scaia
pip install transformers[torch]
pip install sentencepiece wget

```



## Runtime


Running on  3.135.9.103


### How to run

First make sure it is not already rnning:

```bash
ps -ef | grep python
```

Once you are sure it is not already runnning, then run the following:

```bash

source run_service.sh
```


## How to test

```bash
curl -X POST -H "content-type: application/json" -d '{"text": "A central question, “Where is Peng Shuai?”, has represented concern for the star but also points to related questions about the future of tennis in China. "}' http://18.218.29.151:5000/summarizeText
```

