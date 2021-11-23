# Summarizer Service


## Environment

Create a conda enviornment for scaia

```bash
conda create --name scaia python=3.8
conda activate scaia
conda -c anaconda flask
pip install transformers[torch]
pip install sentencepiece

```



## Runtime


Running on 3.16.75.196


```bash
curl -X POST -H "content-type: application/json" -d '{"text": "This is the document to summarize"}' http://3.16.75.196:5000/summarizeText
```
