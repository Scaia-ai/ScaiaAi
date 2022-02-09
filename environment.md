# Initial seatup

```bash
sudo apt update
sudo apt upgrade
eval "$(^Ch-agent -s)"
```


## This repo

```bash
git clone git@github.com:Scaia-ai/ScaiaAi.git

```

# Install anaconda

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
```


# Create Scaia environment

    conda create -n scaia python=3.8 flask -y
    conda activate scaia
    conda install gensim -y
    pip install transformers[torch]
    pip install sentencepiece wget
    pip install fastapi "uvicorn[standard]"

# Open Jupyter Notebook on an AWS server

* On the remote server

    jupyter notebook
    
* On your computer

    ssh -i ~./ssh/ai-advisor.pem -L 8000:localhost:8888 ubuntu@your-remote-ip
  
    - (modify as needed)
    
* Open local browser to

    http://localhost:8000
