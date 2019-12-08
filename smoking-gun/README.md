## Smoking Gun

### running the Program of Ai-Advisor-enron ??
```bash
 sh run_enron.sh
```

### running the Program of Ai-Advisor-freed ??
```bash
 sh run_freed.sh
```


### Steps to run the service as an api
1  conda activate scaia

2  cd ai-advisor

3  python api.py

### Options to use the api 
1  (a) conda activate scaia (b) python api_client.py

2  (a) conda activate scaia (b) jupyter notebook (c) follow steps in api-testing.ipynb 

### Data Storage
All relevant data inluding files in input_data & output_data will be within s3 bucket "ai-advisor-files" for download

### Table of Contents for High Level Script - Ai-Advisor.ipynb:
1  To download Enron Data from Web

2  Process Emails

3  Create Bert Embeddings for Emails

4  Nearest Neighbors - Jaccard

5  Nearest Neighbors - Cosine

6  Doc2Vec

7  Precision & Recall


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

