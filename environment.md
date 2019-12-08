
# Create Scaia environment

    conda create -n scaia
    conda activate scaia
    conda install pytorch torchvision -c pytorch
    conda install gensim

# Open Jupyter Notebook on an AWS server

* On the remote server

    jupyter notebook
    
* On your computer

    ssh -i ~./ssh/ai-advisor.pem -L 8000:localhost:8888 ubuntu@your-remote-ip
  
    - (modify as needed)
    
* Open local browser to

    http://localhost:8000