# Summarizer Service


## Environment

Create a conda enviornment for scaia

```bash
conda create --name scaia python=3.8 flask
conda activate scaia
pip install transformers[torch]
pip install sentencepiece wget
pip install fastapi "uvicorn[standard]"

```



## Runtime


Running on 52.14.40.92


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

# XMap

curl -X POST -H "content-type: application/json" -d '{"text": "A central question, “Where is Peng Shuai?”, has represented concern for the star but also points to related questions about the future of tennis in China. "}' http://52.14.40.92:8000/summarizeText/
```

```bash

# Legal

curl -X POST -H "content-type: application/json" -d '{"text": "On March 5, 2021, the Securities and Exchange Commission charged AT&T, Inc. with repeatedly violating Regulation FD, and three of its Investor Relations executives with aiding and abetting AT&Ts violations, by selectively disclosing material nonpublic information to research analysts. "}' http://52.14.40.92:8000/summarizeTextLegal/

```

```bash

# Pegasus 

curl -X POST -H "content-type: application/json" -d '{"text": "On March 5, 2021, the Securities and Exchange Commission charged AT&T, Inc. with repeatedly violating Regulation FD, and three of its Investor Relations executives with aiding and abetting AT&Ts violations, by selectively disclosing material nonpublic information to research analysts. ", "model" : "google/pegasus-xmap"}' http://52.14.40.92:8000/summarizeTextLegal/

```


## Running on FastAPI

```bash
uvicorn main:app --reload --host 0.0.0.0
```

Yu can see the docs like this:

http://52.14.40.92:8000/docs#


## Docker

We now have a dockerized version of hte applicaiton that runs on port 80.  This may be the fugture of the application

Here is how we run it:

First we should see if the container is already there or running:

```bash
docker ps -a
docker stop <container hash>
docker rm <container hash>
```

```bash
docker build -t "summarizer" .
docker run -d --name summarizer-container -p 80:80 summarizer
```

