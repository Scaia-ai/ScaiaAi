## Smoking Gun

### Running the Program of Ai-Advisor-Mueller ??
```bash
 sh run_mueller.sh
```

### Steps to run the service as an api
1  conda activate scaia

2  cd ai-advisor

3  python api.py

### Options to use the api 
1  (a) conda activate scaia (b) python api_client.py

2  (a) conda activate scaia (b) jupyter notebook (c) follow steps in api-testing.ipynb 

### Data Storage
All relevant data are contained in Mueller-report. Input folder created in scripts

### Table of Contents for High Level Script - Ai-Advisor.ipynb:
1  Split document into Paragraphs

2  Process Emails

3  Create Bert Embeddings for Emails

4  Nearest Neighbors - Manhattan

5  Nearest Neighbors - Cosine

6  Doc2Vec

7  Precision & Recall




