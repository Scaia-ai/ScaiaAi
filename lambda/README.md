
Retrieve an authentication token and authenticate your Docker client to your registry.
Use the AWS CLI:

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 250623887600.dkr.ecr.us-east-1.amazonaws.com
Note: If you receive an error using the AWS CLI, make sure that you have the latest version of the AWS CLI and Docker installed.
Build your Docker image using the following command. For information on building a Docker file from scratch see the instructions here . You can skip this step if your image is already built:

docker build -t bert-lambda .
After the build completes, tag your image so you can push the image to this repository:

docker tag bert-lambda:latest 250623887600.dkr.ecr.us-east-1.amazonaws.com/bert-lambda:latest
Run the following command to push this image to your newly created AWS repository:

docker push 250623887600.dkr.ecr.us-east-1.amazonaws.com/bert-lambda:latest
