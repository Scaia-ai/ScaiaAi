# Docker files
Ai-Advisor-enron

### Overview
This is a docker build file designed to work with miniconda3:4.7.12

### Requirements:
Docker CE
miniconda3

### Build 
```bash
docker-compose build
```
or

```bash
docker build
```

Basically, docker-compose is a better way to use docker than just a docker command.

Docker-compose build, will build individual images, by going into individual service entry in docker-compose.yml.

With docker images, command, we can see all the individual images being saved as well.

The real magic is docker-compose up.

This one will basically create a network of interconnected containers, that can talk to each other with name of container similar to a hostname.


### Run
```bash
docker-compose up -d
```

### Browse docker image
```bash
docker-compose up -d
```
docker-compose run  ai-advisor-enron /bin/sh
```