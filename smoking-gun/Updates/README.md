# Smoking Gun - Enron Emails Example

## Contents of repository

### Bash Scripts 

These two scripts run all of the corresponding python scripts to conduct the tests on the raw Enron Emails and the FreeEed processed emails accordingly. More information on these two are below.

1. run_enron.sh
2. run_freed.sh

### Jupyter Notebooks

These two notebooks run the same tests in the bash scripts as well as the python scripts Ai-Advisor-enron.py and Ai-Advisor-freed.py. The contents of the notebooks are listed below but they run the same tests on different inputs.

3. Ai-Advisor-enron.ipynb
4. Ai-Advisor-freed.ipynb
5. api-testing.ipynb : Tests models and script with api


### Python Scripts

#### Scripts for Running Tests

6. Ai-Advisor-enron.py
7. Ai-Advisor-freed.py

#### API scripts 

8. api.py
9. api-testing.py

### Python Utility Files

These files contain various functions used in the notebooks and the python scripts. 

10. get_enron.py : Downloads email input data
11. get_freed.py : Downloads email input data
12. process_data.py : Processes input text files 
11. document_representations_bert.py : Creates BERT embeddings using functions from bert_embeddings and Pytorch BERT.
12. bert_embeddings.py: Uses Pytorch BERT and transformers package to create BERT embeddings and BERT tokens
13. nn_similarity.py : Contains functions for creating and training nearest neighbor models and calculating distance between BERT embeddings of documents. Also contains functions for predicting closest document to sample.
14. doc2vec_similarity: Contains functions for creating and training doc2vec models and then calulating distance similarity. Also contains functions for predicting closest document to sample.
15. precision_recall.py: Helper functions for calculating precision, recall and F1 score metrics for the 3 models.

---

## Running Ai-Advisor-enron.sh
This program uses the enron emails as downloaded. It processes the emails and creates BERT embeddings and then does nearest neighbors calculations to determine the closest document to each document vector based on cosine and manhattan distances. It also uses the Doc2Vec model to calculate distance as well. Then using a sample document string, a prediction function returns the top documents close to the original.

```bash
 sh run_enron.sh
```

## Running Ai-Advisor-freed.sh
This program takes the enron emails as processed input from FreeEed. It then runs through the same steps as the run_enron script.

```bash
 sh run_freed.sh
```


## Steps to run the service as an api
1  conda activate scaia

2  cd ai-advisor

3  python api.py

## Options to use the api 

1  (a) conda activate scaia (b) python api_client.py

2  (a) conda activate scaia (b) jupyter notebook (c) follow steps in api-testing.ipynb 

## Data Storage

I have repopulated the input_data folder with the needed files and directories. The output_data folder now has all of the embeddings, processed text files, csvs and models.

## Table of Contents for python notebooks

1  Download Enron Data from Web

2  Process Emails

3  Create Bert Embeddings for Emails

4  Nearest Neighbors - Manhattan

5  Nearest Neighbors - Cosine

6  Doc2Vec

7  Precision & Recall

8. Prediction of Closest Documents Given Sample Document


###### To access Jupyter Notebook
For the ".pem" file, to access ec2 contact Mark / Tim / Pavan

1. Using ngrok => ngrok http 8888 # to access jupyter notebooks on web

2. Using terminals => 

    ssh -i "C:\projects\scaia\ai-advisor.pem" ubuntu@ec2-3-14-125-217.us-east-2.compute.amazonaws.com
    
    then go ai-advisor directory
    activate scaia environment
    enter "jupyter notebook"
    
    Open a seperate terminal & enter "ssh -i "C:\projects\scaia\ai-advisor.pem" -L 8000:localhost:8888 ubuntu@ec2-3-14-125-217.us-east-2.compute.amazonaws.com"
    
    after this, jupyter notebook will be serving at => http://localhost:8000/

    and the password would be => scaia

