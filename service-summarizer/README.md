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

```bash
curl -X POST -H "content-type: application/json" -d '{"text": "This is the document to summarize"}' http://3.135.9.103:5000/summarizeText
```

