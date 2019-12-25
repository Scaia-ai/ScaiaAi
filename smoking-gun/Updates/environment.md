## Create environment:

### Linux Setup:
```bash

# This is for setting up the ngrok server for jupyter notebook serving via url
sudo apt update
sudo apt install nodejs
nodejs --version
# [terminal restart]
sudo apt install npm
npm --version 
sudo npm install ngrok -g

```

### Install Miniconda:
```bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# [terminal restart]
Create environment:
conda create --name scaia
```

### To activate the environment:
```bash

conda activate scaia
```

### Install packages:

```bash

conda install python=3.6.2
conda install pandas=0.23.4
conda install scikit-learn=0.21.2
conda install notebook=5.7.0
conda install pytorch=1.0 torchvision cudatoolkit=9.0 -c pytorch
pip install pytorch-pretrained-bert==0.6.2
conda install -c conda-forge jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

pip install gevent==1.4.0
conda install awscli

conda install -c anaconda genism=3.8.0

conda install flask=1.0.2

```

### Only If there’s an issue with jupyter notebook serving:
```bash

pip uninstall tornado
pip install tornado==5.1.1
```


### To Setup AWS connection with s3
```bash
aws configure
aws s3 ls s3://ai-advisor-files # checking s3 connection

# to move files from s3 to ec2 & vice versa
aws s3 cp email_texts.csv s3://ai-advisor-files # moving file to s3 
aws s3 cp s3://ai-advisor-files/email_texts.csv email_texts.csv # s3 to ec2
# reference: http://codeomitted.com/transfer-files-from-ec2-to-s3/
```

### To Setup Jupyter Notebook:
```bash

jupyter notebook --generate-config

	set NotebookApp.allow_remote_access=True

jupyter notebook password # password set => scaia

```
